"""CompLLM revision: run bitsandbytes NF4 checkpoints without bitsandbytes.

Why: on the collection machine Windows Application Control blocks the
bitsandbytes native library, so `load_in_4bit` cannot run. PyTorch itself
ships a 4-bit weight-only matmul (`aten._weight_int4pack_mm`, the tinygemm
kernel), which is part of torch's own CUDA library and loads normally.

What: read an NF4 checkpoint (as published by Unsloth for each model in the
open-weight panel), dequantise every quantised linear layer in plain PyTorch,
re-quantise it to asymmetric int4 with group size 128, and pack it for the
tinygemm kernel. Everything else (embeddings, norms, biases, lm_head) loads
as bfloat16. The model therefore carries two rounds of 4-bit rounding; its
weights differ slightly from the lab's release. That is immaterial to the
study, because the same weights both write a model's share of the corpus and
judge it, so "self" is exact.

NF4 layout (bitsandbytes >= 0.41 serialisation, double quantisation on):
  <name>.weight                                 uint8, two 4-bit codes per byte,
                                                first code in the high nibble
  <name>.weight.quant_map                       float32[16], the NF4 codebook
  <name>.weight.absmax                          uint8, one per 64-weight block,
                                                itself 8-bit quantised:
  <name>.weight.nested_quant_map                float32[256]
  <name>.weight.nested_absmax                   float32, one per 256 blocks
  <name>.weight.quant_state.bitsandbytes__nf4   uint8 JSON: shape, blocksize,
                                                nested_blocksize, nested_offset
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

# Group size of the int4 re-pack. 32 is the smallest the kernel accepts and
# keeps the added rounding error lowest (about 8% relative, measured on
# Phi-4-mini layers, against 10% at 128). COMPLLM_INT4=exact instead keeps the
# NF4 codes and dequantises on every forward pass: exact but several times
# slower, used to validate the re-pack.
import os
GROUP = int(os.environ.get("COMPLLM_INT4_GROUP", "32"))
EXACT = os.environ.get("COMPLLM_INT4", "") == "exact"
SUFFIX = ".weight.quant_state.bitsandbytes__nf4"


class NF4Linear(nn.Module):
    """Exact NF4 semantics, dequantised per forward (validation path)."""
    def __init__(self, codes, code, absmax, bs, shape, bias=None):
        super().__init__()
        self.register_buffer("codes", codes)      # uint8, packed
        self.register_buffer("code", code.to(torch.bfloat16))
        self.register_buffer("absmax", absmax.to(torch.bfloat16))
        self.bs, self.shape = bs, shape
        self.in_features, self.out_features = shape[1], shape[0]
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        idx = torch.stack([self.codes >> 4, self.codes & 0x0F], -1).reshape(-1).long()
        n = self.shape[0] * self.shape[1]
        w = (self.code[idx[:n]].reshape(-1, self.bs) * self.absmax[:, None]).reshape(self.shape)
        return nn.functional.linear(x.to(torch.bfloat16), w, self.bias).to(x.dtype)


class CPUEmbedding(nn.Module):
    """Input embedding kept in CPU memory. A lookup is exact wherever it
    runs, and the table (1-2 GB in bf16 for 128k-256k vocabularies) is the
    difference between a 7-9B model fitting an 8 GB card with room for a
    batch or not."""
    def __init__(self, weight: torch.Tensor, dev: str, scale: float = 1.0):
        super().__init__()
        self.weight_cpu = weight.detach().to("cpu", torch.bfloat16)
        self.dev = dev
        # Gemma scales its embeddings by sqrt(hidden_size) inside the
        # embedding module (Gemma2TextScaledWordEmbedding); dropping that
        # leaves the model emitting nothing but blank lines.
        self.scale = scale
        self.num_embeddings, self.embedding_dim = self.weight_cpu.shape

    def forward(self, ids):
        x = nn.functional.embedding(ids.to("cpu"), self.weight_cpu)
        if self.scale != 1.0:
            x = x * torch.tensor(self.scale, dtype=x.dtype)
        return x.to(self.dev)


class Int4Linear(nn.Module):
    def __init__(self, packed, scales_zeros, in_f, out_f, bias=None):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("sz", scales_zeros)
        self.in_features, self.out_features = in_f, out_f
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        y = torch.ops.aten._weight_int4pack_mm(x2, self.packed, GROUP, self.sz)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)


def dequant_nf4(t: dict[str, torch.Tensor], name: str, dev: str) -> torch.Tensor:
    qs = json.loads(bytes(t[name + SUFFIX].tolist()).decode())
    bs = int(qs["blocksize"])
    code = t[name + ".weight.quant_map"].to(dev, torch.float32)
    am = t[name + ".weight.absmax"].to(dev)
    if am.dtype == torch.uint8:  # double quantisation
        nbs = int(qs["nested_blocksize"])
        nam = t[name + ".weight.nested_absmax"].to(dev, torch.float32)
        nmap = t[name + ".weight.nested_quant_map"].to(dev, torch.float32)
        am = nmap[am.long()] * nam.repeat_interleave(nbs)[: am.numel()] + float(qs["nested_offset"])
    else:
        am = am.float()
    p = t[name + ".weight"].to(dev).reshape(-1)
    idx = torch.stack([p >> 4, p & 0x0F], dim=-1).reshape(-1).long()
    shape = qs["shape"]
    n = shape[0] * shape[1]
    w = code[idx[:n]] * am.repeat_interleave(bs)[:n]
    return w.reshape(shape)


def pack_int4(w: torch.Tensor, dev: str = "cuda:0",
              rows: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric round-to-nearest int4, groups of GROUP along the input dim,
    in the layout aten._weight_int4pack_mm expects. Works through `rows`
    output rows at a time: Gemma 2's 256k-row output layer would otherwise
    need several 3.7 GB float32 temporaries, which on an 8 GB card spills
    into shared system memory and can take the whole machine down."""
    n, k = w.shape
    nib, szs = [], []
    for r0 in range(0, n, rows):
        wg = w[r0:r0 + rows].to(dev).float().reshape(-1, k // GROUP, GROUP)
        mn, mx = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-8) / 15
        q = ((wg - mn) / scale).round().clamp(0, 15).to(torch.uint8).reshape(-1, k)
        nib.append((q[:, ::2] << 4) | q[:, 1::2])
        zero_mid = mn + 8 * scale  # tinygemm: w = (q - 8) * scale + zero_mid
        szs.append(torch.cat([scale, zero_mid], dim=-1).to(torch.bfloat16))
        del wg, mn, mx, scale, q, zero_mid
    packed = torch.ops.aten._convert_weight_to_int4pack(torch.cat(nib), 8)
    sz = torch.cat(szs).reshape(n, k // GROUP, 2)
    return packed, sz.transpose(0, 1).contiguous()


class LazyTensors:
    """The checkpoint's tensors, read from disk one at a time on access.
    Reading them all up front held a whole 6 GB checkpoint in RAM."""
    def __init__(self, files):
        from safetensors import safe_open
        self.fh = [safe_open(str(f), framework="pt") for f in files]
        self.where = {k: h for h in self.fh for k in h.keys()}

    def __contains__(self, k):
        return k in self.where

    def __getitem__(self, k):
        return self.where[k].get_tensor(k)

    def get(self, k, default=None):
        return self[k] if k in self.where else default

    def keys(self):
        return self.where.keys()

    def __iter__(self):
        return iter(self.where)

    def shape(self, k):
        return self.where[k].get_slice(k).get_shape()


def local_dir(repo: str) -> Path:
    """Where the checkpoint lives: $COMPLLM_MODELS/<repo name> (default
    ~/models), filled by a resumable download, else the Hugging Face cache."""
    import os
    base = Path(os.environ.get("COMPLLM_MODELS", Path.home() / "models"))
    d = base / repo.split("/")[-1]
    if (d / "config.json").exists():
        return d
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, local_files_only=True))


def load(repo: str, dev: str = "cuda:0"):
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    path = local_dir(repo)
    config = AutoConfig.from_pretrained(path)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

    tensors = LazyTensors(sorted(path.glob("*.safetensors")))

    quant = sorted(k[: -len(SUFFIX)] for k in tensors if k.endswith(SUFFIX))
    used = set()
    for name in quant:
        w = dequant_nf4(tensors, name, dev)
        mod = model.get_submodule(name)
        bias = tensors.get(name + ".bias")
        if bias is not None:
            bias = bias.to(dev, torch.bfloat16)
            used.add(name + ".bias")
        if EXACT:
            qs = json.loads(bytes(tensors[name + SUFFIX].tolist()).decode())
            bs = int(qs["blocksize"])
            absmax = (w.reshape(-1, bs).abs().amax(-1))  # recovered block scales
            code = tensors[name + ".weight.quant_map"].to(dev)
            new = NF4Linear(tensors[name + ".weight"].to(dev).reshape(-1), code, absmax,
                            bs, list(w.shape), bias)
        else:
            packed, sz = pack_int4(w)
            new = Int4Linear(packed, sz, mod.in_features, mod.out_features, bias)
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, new)
        used.update(k for k in tensors if k.startswith(name + ".weight"))
        del w

    for k in list(tensors.keys()):
        if k in used:
            continue
        mod_name, _, attr = k.rpartition(".")
        try:
            mod = model.get_submodule(mod_name)
        except AttributeError:
            continue
        shape = tensors.shape(k)
        # A big input-embedding table (Gemma 2's is 1.8 GB) never visits the
        # GPU: CPUEmbedding serves lookups from RAM, and a tied output layer
        # is packed from it in chunks below.
        on = "cpu" if (k.endswith("embed_tokens.weight")
                       and shape[0] * shape[1] * 2 > 1.5e9) else dev
        setattr(mod, attr, nn.Parameter(tensors[k].to(on, torch.bfloat16),
                                        requires_grad=False))

    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = head is not None and head.weight.is_meta
    if tied:
        head.weight = emb.weight  # tied embeddings (Gemma 2, Phi-4-mini)
    big = emb.weight.numel() * 2 > 1.5e9
    if tied and big and not EXACT:
        # Gemma 2's 256k x 3584 table: the output layer gets the same int4
        # re-pack as every other linear layer, the input side stays exact.
        packed, sz = pack_int4(emb.weight.data, dev)
        model.set_output_embeddings(Int4Linear(packed, sz, emb.weight.shape[1],
                                               emb.weight.shape[0]))
    if not tied or big:
        scale = float(getattr(emb, "scalar_embed_scale", 1.0))
        model.set_input_embeddings(CPUEmbedding(emb.weight.data, dev, scale))
        del emb

    # Buffers built in __init__ (rotary frequencies) are still on meta.
    for name, m in list(model.named_modules()):
        if any(b.is_meta for b in m.buffers(recurse=False)):
            fresh = type(m)(config=getattr(m, "config", config), device=dev)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, fresh)
    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if left:
        raise RuntimeError(f"unloaded tensors: {left[:8]}")
    try:
        model.generation_config = GenerationConfig.from_pretrained(path)
    except OSError:
        pass
    model.eval()
    del tensors
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return model
