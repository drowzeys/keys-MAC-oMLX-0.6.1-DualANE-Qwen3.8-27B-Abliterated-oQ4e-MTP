#!/usr/bin/env python3
"""Prompt-processing benchmark for oMLX: PP proxy = prompt_tokens / TTFT."""
import argparse, hashlib, json, sys, time, urllib.request

URL = "http://127.0.0.1:11500/v1"
import os
MODEL = os.environ.get("BENCH_MODEL", "qwen38-27b-oq4e")

CORPUS = (
    "The Apple silicon unified memory architecture removes the discrete copy between host and device, "
    "which changes how inference engines schedule prefill and decode. A prefill pass is compute bound at "
    "long context, while token generation is bandwidth bound and rarely saturates the matrix units. "
    "Engineers therefore split fixed-shape blocks across accelerators, keeping the recurrent state, the "
    "normalisation layers, and the output projection on a single device to avoid synchronisation stalls. "
)


def build_prompt(tok, n_tokens, marker):
    text = marker + "\n" + CORPUS * (n_tokens // 40 + 8)
    ids = tok.encode(text)
    while len(ids) < n_tokens:
        text += CORPUS
        ids = tok.encode(text)
    return tok.decode(ids[:n_tokens])


def post_stream(prompt, max_tokens, n_prompt_tokens=None):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    out, usage, ntok = [], None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = (chunk.get("choices") or [{}])[0]
            piece = ch.get("text") or ""
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out.append(piece)
                ntok += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    total = time.perf_counter() - t0
    if usage is None or usage.get("prompt_tokens") is None:
        usage = dict(usage or {})
        usage["prompt_tokens"] = n_prompt_tokens
    return {"ttft": ttft, "total": total, "text": "".join(out),
            "stream_tokens": ntok, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)

    results = []
    for n in a.sizes:
        # identity run first on a cold, path-independent prompt (same text on both paths)
        ident = build_prompt(tok, n, "IDENTITY-CHECK-FIXED-PROMPT")
        r = post_stream(ident, a.gen, n)
        rec = {"tag": a.tag, "size": n, "kind": "identity",
               "ttft": r["ttft"], "total": r["total"],
               "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
               "cached": ((r["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
               "sha256": hashlib.sha256(r["text"].encode()).hexdigest()[:16],
               "text_head": r["text"][:120]}
        rec["pp"] = rec["prompt_tokens"] / r["ttft"] if r["ttft"] else None
        results.append(rec); print(json.dumps(rec), flush=True)

        for i in range(a.repeats):
            p = build_prompt(tok, n, f"RUN-{a.tag}-{n}-{i}-UNIQUE-PREFIX-{i*7919}")
            r = post_stream(p, a.gen, n)
            u = r["usage"] or {}
            rec = {"tag": a.tag, "size": n, "kind": "timed", "rep": i,
                   "ttft": r["ttft"], "total": r["total"],
                   "prompt_tokens": u.get("prompt_tokens"),
                   "completion_tokens": u.get("completion_tokens") or r["stream_tokens"],
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
            rec["pp"] = rec["prompt_tokens"] / r["ttft"] if r["ttft"] else None
            gen_t = r["total"] - (r["ttft"] or 0)
            rec["decode_tps"] = (rec["completion_tokens"] - 1) / gen_t if gen_t > 0 else None
            results.append(rec); print(json.dumps(rec), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
