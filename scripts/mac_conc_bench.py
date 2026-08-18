#!/usr/bin/env python3
"""Mac oMLX concurrency bench: prose / coding / edit at C=1..16."""
import argparse, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:11500/v1/completions"
MODEL = os.environ.get("BENCH_MODEL", "qwen38-27b-ablit")

CLASS = """class InventoryLedger:
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
"""

TASKS = {
    "prose": "Explain in about 400 words why token generation on a large language model is bound "
             "by memory bandwidth rather than arithmetic throughput, and what that implies for batching.",
    "coding": "Write a complete Python module implementing an LRU cache with a TTL per entry, "
              "thread-safe, with docstrings and three unit tests. Output only code.",
    "edit": "Here is a Python class:\n\n" + CLASS +
            "\nRe-emit the entire class unchanged, adding one method `low_stock(threshold)` returning "
            "a sorted list of SKUs below the threshold. Output only code.",
}


def one(args):
    prompt, cap = args
    body = {"model": MODEL, "prompt": prompt, "max_tokens": cap,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    el = time.perf_counter() - t0
    u = d.get("usage", {})
    return el, u.get("completion_tokens", 0), u.get("prompt_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--out", default="mac_conc.json")
    a = ap.parse_args()

    rows = []
    for task, text in TASKS.items():
        for c in a.concurrency:
            # unique prefix per request so prefix-cache can't fake it
            prompts = [(f"[run {task}-c{c}-{i}-{i*7919}]\n{text}", a.gen) for i in range(c)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                res = list(ex.map(one, prompts))
            wall = time.perf_counter() - t0
            toks = sum(r[1] for r in res)
            per = [r[1] / r[0] for r in res if r[0] > 0]
            rec = {"task": task, "concurrency": c, "wall_s": round(wall, 2),
                   "total_tokens": toks,
                   "aggregate_tps": round(toks / wall, 1),
                   "per_request_tps": round(sum(per) / len(per), 1) if per else None}
            rows.append(rec)
            print(json.dumps(rec), flush=True)
        print("", flush=True)
    with open(a.out, "w") as f:
        json.dump({"model": MODEL, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
