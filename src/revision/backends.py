"""CompLLM revision: model backends.

Two backends behind one small interface, so every condition runs the same
code on either panel:

  chat(convs, max_new_tokens, temperature)  -> list of completions
  choice_probs(convs, options)              -> P(option) at the first answer
                                               token (HF only; the OpenRouter
                                               backend asks for top logprobs
                                               and falls back to parsing text)
  score(prefixes, continuations)            -> per-token log-probabilities of
                                               each continuation (HF only)

A "conv" is a list of {"role", "content"} messages with user and assistant
turns only. Mistral v0.3 and Gemma 2 templates reject a system role, so no
condition uses one.

The HF backend holds one model on the GPU at a time. Runners loop over
judges on the outside and over items on the inside for that reason.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    MAX_RETRIES, OPENROUTER_URL, REQUEST_TIMEOUT, RETRY_BASE_DELAY, api_key,
)

# Tokens of context a batch may hold, prompt plus generation. Sized for an
# 8 GB card with a 4-bit 7-9B model; lower it if you see CUDA OOM.
TOKEN_BUDGET = int(os.environ.get("COMPLLM_TOKEN_BUDGET", "12000"))
PREFILL_BUDGET = int(os.environ.get("COMPLLM_PREFILL_BUDGET", "4096"))
PREFILL_CHUNK = int(os.environ.get("COMPLLM_PREFILL_CHUNK", "512"))


# ---------------------------------------------------------------------------
# Local Hugging Face models
# ---------------------------------------------------------------------------
class HFBackend:
    def __init__(self, repo: str, quant: str = "4bit"):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)

        self.torch = torch
        self.repo = repo
        # Cap the caching allocator below the card's memory. On Windows (WDDM)
        # an allocation past physical VRAM silently spills into shared system
        # RAM, which is many times slower and drained the laptop's memory
        # until a run had to be killed; with the cap it is an OOM instead.
        # The headroom also covers what the cap does not count (the CUDA
        # context, cuBLAS workspaces): at 0.45 GB Llama still spilled.
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            max(0.5, (total - float(os.environ.get("COMPLLM_VRAM_HEADROOM", 1.1e9))) / total))
        src = repo
        if "bnb-4bit" in repo:
            import int4
            src = str(int4.local_dir(repo))
        self.tok = AutoTokenizer.from_pretrained(src)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if "bnb-4bit" in repo:
            # NF4 checkpoints are repacked for torch's own int4 kernel; see
            # int4.py for why bitsandbytes is not used.
            import int4
            self.model = int4.load(repo)
        else:
            kw = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
            if quant == "4bit":
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(repo, **kw)
            self.model.eval()
        self.budget = self._token_budget()
        self.stop_ids = self._stop_ids()

    def _stop_ids(self) -> list[int]:
        """Every token that ends an assistant turn. Gemma 2's generation
        config lists only <eos>, not <end_of_turn>, so without this it writes
        past the end of its answer until the length cap."""
        ids = set()
        eos = self.model.generation_config.eos_token_id
        ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        for t in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "<|end|>", "</s>",
                  "<|endoftext|>", "<eos>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != self.tok.unk_token_id and tid >= 0:
                ids.add(tid)
        return sorted(ids)

    def _token_budget(self) -> int:
        """Tokens a batch may hold: free GPU memory after the weights, over
        the KV cache per token (with headroom for activations and logits).
        COMPLLM_TOKEN_BUDGET overrides."""
        if os.environ.get("COMPLLM_TOKEN_BUDGET"):
            return TOKEN_BUDGET
        c = self.model.config
        c = getattr(c, "text_config", c)
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        kv = 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * 2  # bytes/token
        free, _ = self.torch.cuda.mem_get_info()
        self.vocab = getattr(c, "vocab_size", 32000)
        budget = max(2000, min(64000, int((free - 1.0e9) / (kv * 1.25))))
        print(f"    [{self.repo.split('/')[-1]}] free {free / 1e9:.1f} GB, "
              f"KV {kv / 1e3:.0f} kB/token, batch budget {budget} tokens", flush=True)
        return budget

    def close(self) -> None:
        # HF modules hold reference cycles; without a collection the weights
        # outlive `del` and the next model loads into a full card.
        import gc
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()

    # -- helpers -----------------------------------------------------------
    def _ids(self, conv: list[dict], gen_prompt: bool = True) -> list[int]:
        out = self.tok.apply_chat_template(conv, add_generation_prompt=gen_prompt,
                                           tokenize=True)
        # transformers 5 returns a BatchEncoding for some templates.
        if hasattr(out, "input_ids") or isinstance(out, dict):
            out = out["input_ids"]
        return list(out)

    def _batches(self, lengths: list[int], extra: int,
                 budget: int | None = None) -> list[list[int]]:
        """Group indices, longest first, so each batch fits the budget: a
        linear token budget (weights' activations, KV cache) and a quadratic
        one, because with left padding and a mask the prefill attention
        scores are materialised, batch x heads x length^2 of them, which for
        a 3k-token lineup is over a gigabyte at batch 2."""
        budget = budget or self.budget
        c = getattr(self.model.config, "text_config", self.model.config)
        heads = c.num_attention_heads
        att_budget = float(os.environ.get("COMPLLM_ATTN_BYTES", 0.3e9))
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            L = max(cur_max, lengths[i] + extra)
            need = L * (len(cur) + 1)
            att = (len(cur) + 1) * heads * L * L * 4  # fp32 scores and softmax
            if cur and (need > budget or att > att_budget):
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, lengths[i] + extra)
        if cur:
            batches.append(cur)
        return batches

    def _left_pad(self, seqs: list[list[int]]):
        torch = self.torch
        n = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, n - len(s):] = torch.tensor(s)
            att[r, n - len(s):] = 1
        return ids.to("cuda:0"), att.to("cuda:0")

    # -- interface ---------------------------------------------------------
    def chat(self, convs: list[list[dict]], max_new_tokens: int,
             temperature: float = 0.0, top_p: float = 1.0,
             seed: int = 0, progress: str = "") -> list[str]:
        torch = self.torch
        enc = [self._ids(c) for c in convs]
        out: list[str | None] = [None] * len(convs)
        batches = self._batches([len(e) for e in enc], int(0.75 * max_new_tokens))
        t0 = time.time()
        for b_i, b in enumerate(batches):
            ids, att = self._left_pad([enc[i] for i in b])
            torch.manual_seed(seed + b_i)
            kw = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=self.tok.pad_token_id,
                      eos_token_id=self.stop_ids)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                # Greedy judging: switch off any sampling defaults and the
                # repetition penalty some generation configs carry (Qwen).
                kw.update(do_sample=False, temperature=None, top_p=None,
                          top_k=None, repetition_penalty=1.0)
            with torch.no_grad():
                if len(b) == 1 and ids.shape[1] > PREFILL_CHUNK:
                    # Chunked prefill: feed all but the last prompt token in
                    # pieces into the KV cache, so attention memory is bounded
                    # by one piece, then let generate() continue from it.
                    from transformers import DynamicCache
                    cache = DynamicCache()
                    n = ids.shape[1] - 1
                    for c0 in range(0, n, PREFILL_CHUNK):
                        c1 = min(c0 + PREFILL_CHUNK, n)
                        self.model(input_ids=ids[:, c0:c1], past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    kw["past_key_values"] = cache
                gen = self.model.generate(input_ids=ids, attention_mask=att, **kw)
            for r, i in enumerate(b):
                out[i] = self.tok.decode(gen[r, ids.shape[1]:],
                                         skip_special_tokens=True).strip()
            if progress:
                done = sum(len(x) for x in batches[: b_i + 1])
                steps = gen.shape[1] - ids.shape[1]
                print(f"    {progress} {done}/{len(convs)} batch={len(b)} "
                      f"prompt={ids.shape[1]} steps={steps} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        return out  # type: ignore[return-value]

    def _is_space(self, tid: int) -> bool:
        return not self.tok.decode([tid]).strip()

    def _option_token_ids(self, option: str) -> list[int]:
        """The tokens that begin each spelling of an option. A tokenizer that
        splits " 7" into a bare space token and "7" (all four in the panel do,
        for digits) would otherwise put that space token in every option, so
        whitespace-only tokens are skipped: the first token that carries the
        option counts."""
        ids = set()
        for variant in (option, " " + option, option.lower(),
                        " " + option.lower(), option.upper()):
            for tid in self.tok(variant, add_special_tokens=False)["input_ids"]:
                if not self._is_space(tid):
                    ids.add(tid)
                    break
        return sorted(ids)

    def _next_probs(self, enc: list[list[int]]):
        """Next-token distribution after each sequence (CPU, float32)."""
        torch = self.torch
        out = [None] * len(enc)
        # A prefill-only pass needs activations, not a KV cache, so the
        # generation budget overestimates what fits; cap it.
        for b in self._batches([len(e) for e in enc], 1, budget=min(self.budget, PREFILL_BUDGET)):
            ids, att = self._left_pad([enc[i] for i in b])
            with torch.no_grad():
                if len(b) == 1 and ids.shape[1] > PREFILL_CHUNK:
                    # Chunked prefill, as in chat(): attention memory is
                    # bounded by one chunk instead of the whole conversation.
                    from transformers import DynamicCache
                    cache, n = DynamicCache(), ids.shape[1]
                    for c0 in range(0, n, PREFILL_CHUNK):
                        c1 = min(c0 + PREFILL_CHUNK, n)
                        out_c = self.model(input_ids=ids[:, c0:c1], past_key_values=cache,
                                           use_cache=True, logits_to_keep=1)
                    logits = out_c.logits[:, -1, :]
                    del cache
                else:
                    logits = self.model(input_ids=ids, attention_mask=att,
                                        logits_to_keep=1).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            for r, i in enumerate(b):
                out[i] = probs[r]
        return out

    def choice_probs(self, convs: list[list[dict]], options: list[str],
                     assistant_prefix: str = "") -> list[dict]:
        """Next-token probability of each option's first token.

        Returns, per conv, {option: p} renormalised over the options, plus
        "_mass": the raw probability the options held before renormalising,
        which says whether the model was actually answering in the format.
        If a judge's most likely next token is bare whitespace (Mistral
        answers " 7" as a space token, then "7"), that token is appended and
        the options are read one position later; "_stepped" records it.
        """
        opt_ids = {o: self._option_token_ids(o) for o in options}
        enc = []
        for c in convs:
            ids = self._ids(c)
            if assistant_prefix:
                ids += self.tok(assistant_prefix, add_special_tokens=False)["input_ids"]
            enc.append(ids)
        probs = self._next_probs(enc)
        step = [i for i, p in enumerate(probs) if self._is_space(int(p.argmax()))]
        if step:
            again = self._next_probs([enc[i] + [int(probs[i].argmax())] for i in step])
            for i, p in zip(step, again):
                probs[i] = p
        res: list[dict | None] = [None] * len(convs)
        for i, p in enumerate(probs):
            raw = {o: float(p[opt_ids[o]].sum()) for o in options}
            mass = sum(raw.values())
            d = {o: (raw[o] / mass if mass > 0 else float("nan")) for o in options}
            d["_mass"] = mass
            d["_stepped"] = i in step
            res[i] = d
        return res  # type: ignore[return-value]

    def score(self, prefixes: list[list[dict]], continuations: list[str]) -> list[list[float]]:
        """Log-probability of every token of each continuation.

        The prefix is a conversation ending in a user turn; the continuation
        is scored as the assistant's reply to it, which is how the judge
        would have produced it. An empty prefix scores the text with only the
        tokenizer's BOS token before it (unconditional).
        """
        torch = self.torch
        seqs, starts = [], []
        for pre, cont in zip(prefixes, continuations):
            if pre:
                p_ids = self._ids(pre)
            else:
                p_ids = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
            c_ids = self.tok(cont, add_special_tokens=False)["input_ids"]
            seqs.append(p_ids + c_ids)
            starts.append(len(p_ids))
        out: list[list[float] | None] = [None] * len(seqs)
        # Full-sequence logits would be (tokens x vocabulary): over 1 GB for
        # a 1k-token batch with a 152k vocabulary. Run the body once, then the
        # output layer and log-softmax on slices of positions.
        head = self.model.get_output_embeddings()
        cap = getattr(getattr(self.model.config, "text_config", self.model.config),
                      "final_logit_softcapping", None)
        SLICE = 256
        for b in self._batches([len(s) for s in seqs], 0,
                               budget=min(PREFILL_BUDGET, self.budget)):
            n = max(len(seqs[i]) for i in b)
            ids = torch.full((len(b), n), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(b), n), dtype=torch.long)
            for r, i in enumerate(b):  # right padding: positions stay aligned
                ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
                att[r, : len(seqs[i])] = 1
            ids, att = ids.to("cuda:0"), att.to("cuda:0")
            with torch.no_grad():
                hid = self.model.model(input_ids=ids, attention_mask=att).last_hidden_state
                for r, i in enumerate(b):
                    # With no prefix and no BOS token (Qwen) the first token
                    # has nothing to condition on and is not scored.
                    s0, e = max(starts[i], 1), len(seqs[i])
                    lps = []
                    for c0 in range(s0 - 1, e - 1, SLICE):
                        c1 = min(c0 + SLICE, e - 1)
                        lg = head(hid[r, c0:c1]).float()
                        if cap:
                            lg = torch.tanh(lg / cap) * cap
                        lp = torch.log_softmax(lg, dim=-1)
                        tgt = ids[r, c0 + 1:c1 + 1]
                        lps.append(lp.gather(1, tgt[:, None]).squeeze(1))
                    out[i] = torch.cat(lps).tolist()
            del hid
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# OpenRouter (frontier panel). Written to the same interface; not run in the
# revision so far because the key on file was revoked.
# ---------------------------------------------------------------------------
class OpenRouterBackend:
    def __init__(self, slug: str, raw_dir: Path, max_tokens: int = 8000):
        import requests
        self.requests = requests
        self.slug = slug
        self.key = api_key()
        self.raw_dir = raw_dir
        self.max_tokens = max_tokens
        raw_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        pass

    def _call(self, conv: list[dict], max_tokens: int, temperature: float,
              logprobs: bool = False) -> dict:
        payload = {"model": self.slug, "messages": conv,
                   "temperature": temperature, "max_tokens": max_tokens}
        if logprobs:
            payload.update(logprobs=True, top_logprobs=10)
        last = None
        for attempt in range(MAX_RETRIES):
            try:
                r = self.requests.post(
                    OPENROUTER_URL, json=payload, timeout=REQUEST_TIMEOUT,
                    headers={"Authorization": f"Bearer {self.key}",
                             "X-Title": "CompLLM revision"})
                if r.status_code in (408, 409, 429, 500, 502, 503, 504, 520, 522, 524):
                    raise RuntimeError(f"transient HTTP {r.status_code}")
                r.raise_for_status()
                body = r.json()
                if body.get("choices"):
                    return body
                raise RuntimeError("no choices")
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(RETRY_BASE_DELAY * 2 ** attempt + random.uniform(0, 2))
        raise RuntimeError(f"exhausted retries: {last}")

    def chat(self, convs, max_new_tokens, temperature=0.0, top_p=1.0, seed=0,
             progress="", ids=None) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        from config import CONCURRENCY

        def one(k):
            body = self._call(convs[k], max(max_new_tokens, self.max_tokens), temperature)
            if ids:
                (self.raw_dir / f"{ids[k]}.json").write_text(
                    json.dumps({"request_model": self.slug, "messages": convs[k],
                                "response": body}, indent=2), encoding="utf-8")
            return body["choices"][0]["message"]["content"] or ""

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            return list(pool.map(one, range(len(convs))))

    def choice_probs(self, convs, options, assistant_prefix="", ids=None) -> list[dict]:
        """Top-logprob probabilities where the provider returns them, else a
        one-hot from the parsed text. "_mass" is NaN when parsed from text."""
        from concurrent.futures import ThreadPoolExecutor
        from config import CONCURRENCY

        def one(k):
            # None of the five frontier routes returned logprobs on 5 Oct 2026,
            # and OpenAI rejects the request for reasoning models, so the
            # answer is read from the text.
            body = self._call(convs[k], self.max_tokens, 0.0, logprobs=False)
            if ids:
                (self.raw_dir / f"{ids[k]}.json").write_text(
                    json.dumps({"request_model": self.slug, "messages": convs[k],
                                "response": body}, indent=2), encoding="utf-8")
            text = (body["choices"][0]["message"]["content"] or "").strip()
            lp = ((body["choices"][0].get("logprobs") or {}).get("content") or [])
            if lp:
                top = {t["token"].strip().lower(): math.exp(t["logprob"])
                       for t in lp[0].get("top_logprobs", [])}
                raw = {o: sum(p for t, p in top.items() if t == o.lower()) for o in options}
                mass = sum(raw.values())
                if mass > 0:
                    d = {o: raw[o] / mass for o in options}
                    d["_mass"] = mass
                    d["_text"] = text
                    return d
            first = text.split()[0].strip(".,:;!*\"'").lower() if text else ""
            d = {o: float(first == o.lower()) for o in options}
            d["_mass"] = float("nan")
            d["_text"] = text
            return d

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            return list(pool.map(one, range(len(convs))))

    def score(self, prefixes, continuations):
        raise NotImplementedError("closed models do not expose prompt log-probabilities")


def make_backend(panel, name: str):
    if panel.backend == "hf":
        return HFBackend(panel.model_ids[name])
    from config import MAX_TOKENS, MAX_TOKENS_BY_JUDGE
    # The submitted study's per-judge limits: Gemini's reasoning tokens count
    # against the cap, and at 8,000 it failed to return JSON.
    return OpenRouterBackend(panel.model_ids[name], panel.results_dir / "raw" / name,
                             max_tokens=MAX_TOKENS_BY_JUDGE.get(name, MAX_TOKENS))
