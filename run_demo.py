"""Generate the project deliverables.

Outputs (written to ./outputs):
  sense_inventory.csv  - all WordNet senses + relations for the target words
  wsd_results.csv      - per-word predictions for the demo text (extended Lesk)
  evaluation.csv       - per-sentence comparison of Lesk vs. MFS on the test set
  evaluation_summary.json
"""
import json
import os

import pandas as pd

import wsd_core as w
from gold_data import DEMO_TEXT, GOLD_ITEMS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

TARGETS = ["bank", "bass", "bat", "plant", "spring", "mouse", "crane", "match", "interest", "light"]

# 1. Sense inventory
inv = w.inventory_frame(TARGETS)
inv.to_csv(os.path.join(OUT, "sense_inventory.csv"), index=False)
print(f"sense_inventory.csv: {len(inv)} senses for {len(TARGETS)} words")
print(inv.groupby("word")["synset"].count().to_string())

# 2. WSD on sample text
res = w.disambiguate_text(DEMO_TEXT, method="lesk", only_ambiguous=True)
res.to_csv(os.path.join(OUT, "wsd_results.csv"), index=False)
print(f"\nwsd_results.csv: {len(res)} ambiguous words disambiguated")
focus = res[res["lemma"].isin(TARGETS)]
print(focus[["sentence", "token", "predicted_synset", "score", "method"]].to_string(index=False))

# 3. Evaluation: Lesk vs MFS baseline
lesk = w.evaluate(GOLD_ITEMS, "lesk")
mfs = w.evaluate(GOLD_ITEMS, "mfs")
rows = []
for a, b in zip(lesk["details"], mfs["details"]):
    rows.append(
        {
            "sentence": a["sentence"],
            "target": a["target"],
            "gold": a["gold_synset"],
            "lesk_pred": a["predicted"],
            "lesk_ok": a["correct"],
            "lesk_method": a["used"],
            "mfs_pred": b["predicted"],
            "mfs_ok": b["correct"],
        }
    )
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(OUT, "evaluation.csv"), index=False)
by_word = ev.groupby("target")[["lesk_ok", "mfs_ok"]].mean().round(2)
summary = {
    "n_sentences": len(ev),
    "lesk_accuracy": round(lesk["accuracy"], 3),
    "mfs_accuracy": round(mfs["accuracy"], 3),
    "lesk_fallback_to_mfs": int((ev["lesk_method"] == "mfs-fallback").sum()),
    "by_word": by_word.to_dict(orient="index"),
}
with open(os.path.join(OUT, "evaluation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nEvaluation:", json.dumps({k: v for k, v in summary.items() if k != "by_word"}))
print(by_word.to_string())
print("\nLesk errors:")
print(ev[~ev["lesk_ok"]][["target", "sentence", "gold", "lesk_pred"]].to_string(index=False))
