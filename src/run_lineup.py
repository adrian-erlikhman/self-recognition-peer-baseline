"""CompLLM: Engine B lineup harness.

Runs the position-counterbalanced blind lineup against OpenRouter.

  one session per (prompt, judge); each yields five slot-level judgments

Every raw completion is archived to raw/<session_id>__<judge>.json so the
analysis is reproducible from stored data even if providers are
non-deterministic. The run is resumable: sessions already present in
results/sessions.jsonl are skipped.

Usage
-----
    python src/check_models.py                 # verify slugs + prices first
    python src/run_lineup.py --dry-run         # print one prompt, estimate cost
    python src/run_lineup.py --limit 2         # 2 prompts x 5 judges = 10 sessions
    python src/run_lineup.py                   # full run
    python src/parse_judgments.py              # -> results/judgments.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CONCURRENCY,
    DATA_DIR,
    EXCLUDE_PROMPTS,
    JUDGE_SLUGS,
    MAX_RETRIES,
    MAX_TOKENS,
    MAX_TOKENS_BY_JUDGE,
    MODELS,
    OPENROUTER_URL,
    RAW_DIR,
    RESPONSES_CSV,
    RESPONSES_SHA256,
    RESULTS_DIR,
    RETRY_BASE_DELAY,
    REQUEST_TIMEOUT,
    SESSIONS_JSONL,
    SLOTS,
    TEMPERATURE,
    api_key,
)
from design import build_design, verify_balance, write_design  # noqa: E402

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Corpus integrity
# ---------------------------------------------------------------------------
def load_corpus(strict_hash: bool = True) -> pd.DataFrame:
    digest = hashlib.sha256(RESPONSES_CSV.read_bytes()).hexdigest()
    if digest != RESPONSES_SHA256:
        msg = (
            f"Corpus hash mismatch.\n"
            f"  expected {RESPONSES_SHA256}\n"
            f"  actual   {digest}\n"
            f"The frozen dataset changed. Either restore it or update "
            f"RESPONSES_SHA256 in config.py and regenerate every number."
        )
        if strict_hash:
            raise SystemExit(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    df = pd.read_csv(RESPONSES_CSV)
    df["prompt_id"] = df["prompt_id"].astype(str)
    missing = set(MODELS) - set(df["model"].unique())
    if missing:
        raise SystemExit(f"Corpus is missing responses for: {sorted(missing)}")
    return df


def load_prompt_texts() -> dict[str, str]:
    """Optional. data/prompts.csv with columns prompt_id,prompt_text.

    If absent, the judge sees only the five responses. The committed run was
    collected without it. Adding the file changes what judges see, so it
    belongs to a separate run, not a resume of this one.
    """
    path = DATA_DIR / "prompts.csv"
    if not path.exists():
        return {}
    pdf = pd.read_csv(path)
    return {str(r.prompt_id): str(r.prompt_text) for r in pdf.itertuples()}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
JUDGE_INSTRUCTIONS = """\
Below are five AI-generated responses to the same task, labelled A through E.

Each response was written by one of these five language models:
GPT, Claude, Gemini, Grok, DeepSeek.

For each of the five responses, identify which model you believe wrote it, \
judging only from writing style, structure, and phrasing.

Answer independently for each response. Your five answers do not need to be \
five different models - name whichever model you believe wrote each response, \
even if that means naming the same model more than once.

Reply with JSON only, in exactly this form and nothing else:

{
  "A": {"model": "<one of GPT, Claude, Gemini, Grok, DeepSeek>", "confidence": <1-5>, "reason": "<one short sentence>"},
  "B": {"model": "...", "confidence": <1-5>, "reason": "..."},
  "C": {"model": "...", "confidence": <1-5>, "reason": "..."},
  "D": {"model": "...", "confidence": <1-5>, "reason": "..."},
  "E": {"model": "...", "confidence": <1-5>, "reason": "..."}
}

confidence: 1 = pure guess, 5 = certain."""


def build_session_prompt(
    prompt_id: str,
    slot_to_text: dict[str, str],
    prompt_text: str | None,
) -> str:
    parts = [JUDGE_INSTRUCTIONS]
    if prompt_text:
        parts.append(f"\n--- TASK GIVEN TO ALL FIVE MODELS ---\n{prompt_text.strip()}")
    for slot in SLOTS:
        parts.append(f"\n--- RESPONSE {slot} ---\n{slot_to_text[slot].strip()}")
    parts.append(
        "\n--- END OF RESPONSES ---\n"
        "Now output the JSON object described above, and nothing else."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------
class TransientError(Exception):
    pass


def call_openrouter(slug: str, prompt: str, key: str, max_tokens: int | None = None) -> dict:
    payload = {
        "model": slug,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens or MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "CompLLM lineup",
    }
    r = requests.post(
        OPENROUTER_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
    )
    if r.status_code in (408, 409, 429, 500, 502, 503, 504, 520, 522, 524):
        raise TransientError(f"HTTP {r.status_code}: {r.text[:300]}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:600]}")
    body = r.json()
    if "choices" not in body or not body["choices"]:
        raise TransientError(f"no choices in response: {json.dumps(body)[:300]}")
    return body


def call_with_retries(slug: str, prompt: str, key: str,
                      max_tokens: int | None = None) -> dict:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return call_openrouter(slug, prompt, key, max_tokens)
        except TransientError as exc:
            last = exc
            delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 2)
            time.sleep(delay)
        except requests.RequestException as exc:
            last = exc
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
    raise RuntimeError(f"exhausted {MAX_RETRIES} retries: {last}")


# ---------------------------------------------------------------------------
# Parsing (defensive - judges do not always obey the format)
# ---------------------------------------------------------------------------
# What a judge may put between the slot letter and the model name. The last
# entry is an em dash, written as an escape so this file stays pure ASCII;
# judges do emit them, so it has to stay in the set.
SLOT_SEPARATORS = re.escape(":.-\u2014)")


_CANON = {m.lower(): m for m in MODELS}
_CANON.update(
    {
        "gpt-4o": "GPT", "gpt4o": "GPT", "chatgpt": "GPT", "openai": "GPT",
        "gpt-5": "GPT", "gpt-5.5": "GPT", "gpt5": "GPT", "gpt-4": "GPT",
        "claude sonnet": "Claude", "claude opus": "Claude", "sonnet": "Claude",
        "opus": "Claude", "anthropic": "Claude", "haiku": "Claude",
        "gemini pro": "Gemini", "google": "Gemini", "bard": "Gemini",
        "gemeni": "Gemini",
        "grok": "Grok", "xai": "Grok", "x-ai": "Grok",
        "deep seek": "DeepSeek", "deepseek-v3": "DeepSeek", "deepseek chat": "DeepSeek",
    }
)


def canon_model(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().lower()
    if s in _CANON:
        return _CANON[s]
    for key, val in _CANON.items():
        if key in s:
            return val
    return None


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a completion."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_completion(text: str) -> tuple[dict, bool]:
    """Return ({slot: {model, confidence, reason}}, needs_review)."""
    out: dict[str, dict] = {}
    obj = extract_json(text)
    if obj:
        for slot in SLOTS:
            val = obj.get(slot) or obj.get(slot.lower())
            if isinstance(val, dict):
                model = canon_model(val.get("model") or val.get("Model"))
                conf, reason = val.get("confidence"), val.get("reason")
            else:
                model, conf, reason = canon_model(val), None, None
            if model:
                out[slot] = {"model": model, "confidence": conf, "reason": reason}

    # Fallback: line-oriented "A: Claude" / "A - Claude"
    if len(out) < len(SLOTS):
        for slot in SLOTS:
            if slot in out:
                continue
            m = re.search(
                rf"^\s*\**{slot}\**\s*[{SLOT_SEPARATORS}]\s*\**\s*([A-Za-z0-9 .\-]+)",
                text,
                re.M,
            )
            model = canon_model(m.group(1)) if m else None
            if model:
                out[slot] = {"model": model, "confidence": None, "reason": None}

    return out, len(out) != len(SLOTS)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def completed_sessions() -> set[str]:
    if not SESSIONS_JSONL.exists():
        return set()
    done = set()
    with SESSIONS_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A session that returned but could not be parsed is NOT done -
            # re-running should retry it. Otherwise the Gemini sessions that
            # blew their token budget mid-reasoning would be locked in as
            # permanently empty.
            if rec.get("ok") and not rec.get("needs_review"):
                done.add(rec["session_id"])
    return done


def run_session(session, key: str) -> dict:
    sid = session["session_id"]
    slug = JUDGE_SLUGS[session["judge"]]
    try:
        body = call_with_retries(slug, session["prompt"], key,
                                 MAX_TOKENS_BY_JUDGE.get(session["judge"]))
        text = body["choices"][0]["message"]["content"] or ""
        parsed, needs_review = parse_completion(text)
        usage = body.get("usage", {}) or {}
        rec = {
            "session_id": sid,
            "prompt_id": session["prompt_id"],
            "judge": session["judge"],
            "judge_slug": slug,
            "slot_to_author": session["slot_to_author"],
            "parsed": parsed,
            "needs_review": needs_review,
            "raw_text": text,
            "usage": usage,
            "ok": True,
        }
        (RAW_DIR / f"{sid}.json").write_text(
            json.dumps({"request_model": slug, "response": body}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - one bad session must not kill the run
        rec = {
            "session_id": sid,
            "prompt_id": session["prompt_id"],
            "judge": session["judge"],
            "judge_slug": slug,
            "error": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }

    with _write_lock:
        with SESSIONS_JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print one prompt, estimate cost, exit")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts")
    ap.add_argument("--judges", default="", help="comma-separated subset, e.g. GPT,Claude")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus(strict_hash=not args.allow_hash_mismatch)
    prompt_texts = load_prompt_texts()
    all_prompts = sorted(df["prompt_id"].unique())
    prompts = [p for p in all_prompts if p not in EXCLUDE_PROMPTS]
    dropped = [p for p in all_prompts if p in EXCLUDE_PROMPTS]
    if dropped:
        print(f"excluding prompt(s) {dropped} (see EXCLUDE_PROMPTS in config.py)")
    df = df[df["prompt_id"].isin(prompts)]
    judges = [j.strip() for j in args.judges.split(",") if j.strip()] or list(MODELS)

    design = build_design(prompts, list(MODELS))
    verify_balance(design, n_prompts=len(prompts))
    write_design(design)

    if args.limit:
        keep = set(prompts[: args.limit])
        design = [r for r in design if r["prompt_id"] in keep]
    design = [r for r in design if r["judge"] in judges]

    # Group the flat design into sessions
    by_session: dict[str, dict] = {}
    for r in design:
        s = by_session.setdefault(
            r["session_id"],
            {
                "session_id": r["session_id"],
                "prompt_id": r["prompt_id"],
                "judge": r["judge"],
                "slot_to_author": {},
            },
        )
        s["slot_to_author"][r["slot"]] = r["true_author"]

    text_lookup = {
        (str(row.prompt_id), row.model): row.text for row in df.itertuples()
    }
    sessions = []
    for s in by_session.values():
        slot_to_text = {
            slot: text_lookup[(s["prompt_id"], author)]
            for slot, author in s["slot_to_author"].items()
        }
        s["prompt"] = build_session_prompt(
            s["prompt_id"], slot_to_text, prompt_texts.get(s["prompt_id"])
        )
        sessions.append(s)
    sessions.sort(key=lambda s: (s["prompt_id"], s["judge"]))

    if args.dry_run:
        ex = sessions[0]
        print("=" * 72)
        print(f"EXAMPLE SESSION  {ex['session_id']}   judge={ex['judge']}")
        print(f"slot -> true author: {ex['slot_to_author']}")
        print(f"prompt text shown to judge: {'YES' if prompt_texts else 'NO (data/prompts.csv absent)'}")
        print("=" * 72)
        print(ex["prompt"][:3000])
        print("... [truncated for display]")
        print("=" * 72)
        chars = sum(len(s["prompt"]) for s in sessions)
        in_tok = chars / 4
        out_tok = len(sessions) * 350
        print(f"sessions        : {len(sessions)}")
        print(f"judgments       : {len(sessions) * 5}")
        print(f"est. input tok  : {in_tok:,.0f}")
        print(f"est. output tok : {out_tok:,.0f}")
        print(f"est. cost       : ${in_tok/1e6*4 + out_tok/1e6*14:,.2f} "
              f"(rough, at ~$4/M in, ~$14/M out; check_models.py has real prices)")
        return

    key = api_key()
    done = completed_sessions()
    todo = [s for s in sessions if s["session_id"] not in done]
    print(f"sessions total {len(sessions)} | already done {len(done)} | to run {len(todo)}")
    if not todo:
        print("nothing to do. run:  python src/parse_judgments.py")
        return

    ok = fail = review = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = {pool.submit(run_session, s, key): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec.get("ok"):
                ok += 1
                review += 1 if rec.get("needs_review") else 0
            else:
                fail += 1
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(
                    f"  [{i}/{len(todo)}] ok={ok} fail={fail} needs_review={review} "
                    f"({el:.0f}s)"
                )

    print(f"\ndone in {time.time()-t0:.0f}s - ok={ok} fail={fail} needs_review={review}")
    if fail:
        print("re-run the same command to retry failed sessions (the run is resumable).")
    print("next:  python src/parse_judgments.py")


if __name__ == "__main__":
    main()
