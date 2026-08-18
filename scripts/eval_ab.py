#!/usr/bin/env python3
"""Run oMLX's own accuracy benchmarks against a served model over HTTP.

Reuses omlx.eval dataset loaders + scoring; supplies a duck-typed engine whose
chat() posts to /v1/chat/completions, so no admin session is needed.
"""
import argparse, asyncio, json, urllib.request
from types import SimpleNamespace

URL = "http://127.0.0.1:11500/v1/chat/completions"


class HTTPEngine:
    is_external_api = False
    model_type = "qwen3_5"

    def __init__(self, model_id, timeout=3600, thinking=False, effort=None):
        self.model_id = model_id
        self.timeout = timeout
        self.thinking = thinking
        self.effort = effort

    async def chat(self, messages, **kwargs):
        body = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "chat_template_kwargs": kwargs.get("chat_template_kwargs")
                                    or {"enable_thinking": self.thinking},
        }
        if self.effort:
            body["reasoning_effort"] = self.effort
        for k in ("top_p", "top_k", "min_p", "repetition_penalty"):
            if kwargs.get(k) is not None:
                body[k] = kwargs[k]

        def _post():
            req = urllib.request.Request(
                URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)

        try:
            d = await asyncio.to_thread(_post)
            text = d["choices"][0]["message"].get("content") or ""
        except Exception as e:                     # keep the run alive
            print(f"  [request failed: {type(e).__name__}: {e}]", flush=True)
            text = ""
        return SimpleNamespace(text=text)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--thinking-budget", type=int, default=17000)
    ap.add_argument("--effort", default=None, help="reasoning_effort, e.g. xhigh")
    ap.add_argument("--benchmarks", nargs="+", required=True,
                    help="name:samples pairs, e.g. mmlu:200 gsm8k:120")
    a = ap.parse_args()

    from omlx.eval import BENCHMARKS
    import omlx.eval.base as _base
    if a.thinking:
        _base.THINKING_MIN_TOKENS = a.thinking_budget
        _base.THINKING_MAX_TOKENS = max(_base.THINKING_MAX_TOKENS, a.thinking_budget)
        print(f"thinking budget: {_base.THINKING_MIN_TOKENS} tokens, effort={a.effort}", flush=True)

    engine = HTTPEngine(a.model_id, thinking=a.thinking, effort=a.effort)
    out = {"tag": a.tag, "model_id": a.model_id, "thinking": a.thinking,
           "budget": a.thinking_budget if a.thinking else None, "effort": a.effort, "results": {}}

    for spec in a.benchmarks:
        name, _, n = spec.partition(":")
        n = int(n or 0)
        bench = BENCHMARKS[name]()
        items = await bench.load_dataset(sample_size=n)
        print(f"[{a.tag}] {name}: {len(items)} items", flush=True)

        async def progress(cur, tot, _n=name):
            if cur % 25 == 0 or cur == tot:
                print(f"  {_n} {cur}/{tot}", flush=True)

        res = await bench.run(engine, items, on_progress=progress,
                              batch_size=a.batch_size,
                              sampling_kwargs={"temperature": 0},
                              enable_thinking=a.thinking)
        out["results"][name] = {
            "accuracy": res.accuracy,
            "correct": res.correct_count,
            "total": res.total_questions,
            "seconds": res.time_seconds,
        }
        print(f"[{a.tag}] {name} = {res.accuracy:.1%} "
              f"({res.correct_count}/{res.total_questions}) in {res.time_seconds:.0f}s",
              flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
