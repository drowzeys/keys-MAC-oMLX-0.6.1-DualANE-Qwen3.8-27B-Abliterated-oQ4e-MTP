#!/usr/bin/env python3
"""Lightning MTP sweep: decode throughput by task class + losslessness check."""
import argparse, hashlib, json, time, urllib.request

URL = "http://127.0.0.1:11500/v1"
import os
MODEL = os.environ.get("BENCH_MODEL", "qwen38-27b-oq4e")

FILLER = (
    "Unified memory on Apple silicon removes the discrete host-to-device copy, so prefill and decode "
    "share one allocator. Prefill is compute bound; decode is bandwidth bound and rarely saturates the "
    "matrix units. Schedulers therefore chunk prompts and interleave decode steps to keep both busy. "
)

CODE_BODY = '''class InventoryLedger:
    def __init__(self, warehouse_id, clock):
        self.warehouse_id = warehouse_id
        self.clock = clock
        self._items = {}
        self._audit = []

    def add_item(self, sku, quantity, unit_cost):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        entry = self._items.setdefault(sku, {"quantity": 0, "unit_cost": unit_cost})
        entry["quantity"] += quantity
        self._audit.append((self.clock.now(), "add", sku, quantity))
        return entry["quantity"]

    def remove_item(self, sku, quantity):
        entry = self._items.get(sku)
        if entry is None or entry["quantity"] < quantity:
            raise KeyError(sku)
        entry["quantity"] -= quantity
        self._audit.append((self.clock.now(), "remove", sku, quantity))
        return entry["quantity"]

    def total_value(self):
        return sum(e["quantity"] * e["unit_cost"] for e in self._items.values())

    def audit_trail(self):
        return list(self._audit)
'''

TASKS = {
    "prose": "Explain, in about 300 words, why token generation on a large language model is bound by "
             "memory bandwidth rather than arithmetic throughput, and what that implies for batching.",
    "codeedit": "Here is a Python class:\n\n" + CODE_BODY +
                "\nRe-emit the entire class unchanged, adding one new method `low_stock(threshold)` that "
                "returns a sorted list of SKUs whose quantity is below the threshold. Output only code.",
}

QUALITY = [
    "In three sentences, explain what a gated delta network is and why it helps long-context models.",
    "A train leaves at 14:35 and the journey takes 2 hours 50 minutes. What time does it arrive? Show your reasoning.",
    "Write a Python function that merges two sorted lists into one sorted list, without using sorted().",
    "Name the capital of Australia and one common misconception about it.",
]


def post(prompt, max_tokens, stream=True, temperature=0):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    if not stream:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        return {"ttft": None, "total": time.perf_counter() - t0,
                "text": d["choices"][0]["text"], "usage": d.get("usage", {}), "ntok": None}
    ttft, out, usage, ntok = None, [], None, 0
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
            piece = (chunk.get("choices") or [{}])[0].get("text") or ""
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out.append(piece); ntok += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    return {"ttft": ttft, "total": time.perf_counter() - t0,
            "text": "".join(out), "usage": usage or {}, "ntok": ntok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)

    def pad(task_text, marker):
        pre = f"[{marker}]\n" + FILLER * 400
        ids = tok.encode(pre)
        return tok.decode(ids[:a.ctx]) + "\n\n" + task_text

    results = []
    for name, task in TASKS.items():
        for i in range(a.repeats):
            p = pad(task, f"{a.tag}-{name}-{i}-{i*7919}")
            r = post(p, a.gen)
            gen_t = r["total"] - (r["ttft"] or 0)
            n = r["usage"].get("completion_tokens") or r["ntok"]
            rec = {"tag": a.tag, "task": name, "rep": i,
                   "decode_tps": (n - 1) / gen_t if gen_t > 0 and n else None,
                   "gen_tokens": n, "ttft": r["ttft"], "total": r["total"],
                   "prompt_tokens": r["usage"].get("prompt_tokens"),
                   "cached": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")}
            results.append(rec); print(json.dumps(rec), flush=True)

    if a.quality:
        for qi, q in enumerate(QUALITY):
            r = post(q, 400, stream=False)
            rec = {"tag": a.tag, "kind": "quality", "q": qi,
                   "sha256": hashlib.sha256(r["text"].encode()).hexdigest()[:16],
                   "chars": len(r["text"]), "text": r["text"]}
            results.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "text"}), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
