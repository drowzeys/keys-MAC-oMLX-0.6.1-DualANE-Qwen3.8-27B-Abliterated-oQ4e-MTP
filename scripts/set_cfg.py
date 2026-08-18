import json, pathlib, sys
# usage: set_cfg.py <model_key> <ane on|off> <mtp off|N>
key, ane, mtp = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path.home() / ".omlx/model_settings.json"
d = json.loads(p.read_text())
m = d["models"].setdefault(key, {})
if ane == "on":
    m.update({"qwen35_ane_prefill_enabled": True, "qwen35_ane_prefill_sequence_length": 2048,
              "qwen35_ane_prefill_fraction": 0.53, "qwen35_ane_prefill_max_layers": 64,
              "qwen35_ane_prefill_dual_ane": True, "qwen35_ane_prefill_gdn": False,
              "qwen35_ane_prefill_gdn_max_layers": 0})
else:
    m["qwen35_ane_prefill_enabled"] = False
if mtp == "off":
    m["mtp_enabled"] = False
    m.pop("mtp_num_draft_tokens", None)
else:
    m["mtp_enabled"] = True
    m["mtp_num_draft_tokens"] = int(mtp)
p.write_text(json.dumps(d, indent=2))
print(f"{key}: ane={ane} mtp={mtp}")
