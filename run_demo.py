"""Demo: train, unlearn two ways, verify both against the gold standard.

Run from the project root:
  python3 run_demo.py

It writes runs/report.json and prints a table. No API key, no network,
no downloads. The model is a toy fact model in plain Python.
"""

import copy
import json
import time
from pathlib import Path

from forgetcheck.data import build_dataset
from forgetcheck.model import FactModel, evaluate, train_model
from forgetcheck.unlearn import unlearn_gradient_ascent, unlearn_random_label
from forgetcheck.verify import mia_auc, verdict


def fresh_model(data, seed=0):
    return FactModel(
        n_subjects=len(data.subjects),
        n_relations=len(data.relations),
        n_objects=len(data.objects),
        hidden=24,
        seed=seed,
    )


def check(model, data):
    retain = evaluate(model, data.retain)
    forget = evaluate(model, data.forget)
    unseen = evaluate(model, data.unseen)
    auc = mia_auc(forget["losses"], unseen["losses"])
    result = verdict(forget["accuracy"], retain["accuracy"], auc)
    return {
        "retain_accuracy": round(retain["accuracy"], 3),
        "retain_mean_loss": round(retain["mean_loss"], 3),
        "forget_accuracy": round(forget["accuracy"], 3),
        "forget_mean_loss": round(forget["mean_loss"], 3),
        "unseen_mean_loss": round(unseen["mean_loss"], 3),
        "mia_auc": round(auc, 3),
        "passed": result["passed"],
        "reasons": result["reasons"],
    }


def main():
    started = time.time()
    data = build_dataset()
    base = fresh_model(data)
    train_model(base, data.retain + data.forget, epochs=200, lr=1.0)

    gold = fresh_model(data)
    train_model(gold, data.retain, epochs=200, lr=1.0)

    by_random_label = copy.deepcopy(base)
    info_random = unlearn_random_label(by_random_label, data.retain, data.forget)

    by_ascent = copy.deepcopy(base)
    info_ascent = unlearn_gradient_ascent(by_ascent, data.retain, data.forget)

    rows = [
        ("base (trained on retain plus forget)", check(base, data), {}),
        ("gold (retrained on retain only)", check(gold, data), {}),
        ("random label unlearning", check(by_random_label, data), info_random),
        ("gradient ascent unlearning", check(by_ascent, data), info_ascent),
    ]
    report = {
        "dataset": {
            "retain": len(data.retain),
            "forget": len(data.forget),
            "unseen": len(data.unseen),
            "objects": len(data.objects),
            "chance_accuracy": round(1.0 / len(data.objects), 3),
        },
        "seconds": round(time.time() - started, 2),
        "results": [
            {"name": name, "info": info, **result} for name, result, info in rows
        ],
    }
    out = Path(__file__).parent / "runs" / "report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("ForgetCheck demo")
    print("chance accuracy: %.3f" % report["dataset"]["chance_accuracy"])
    print("")
    for item in report["results"]:
        print(item["name"])
        print(
            "  retain acc %.3f  forget acc %.3f  forget loss %.2f"
            "  unseen loss %.2f  MIA AUC %.3f  verdict %s"
            % (
                item["retain_accuracy"],
                item["forget_accuracy"],
                item["forget_mean_loss"],
                item["unseen_mean_loss"],
                item["mia_auc"],
                "PASS" if item["passed"] else "FAIL",
            )
        )
        if item["reasons"]:
            print("  why: " + "; ".join(item["reasons"]))
        if item["info"]:
            print("  unlearning epochs used: %s" % item["info"]["epochs_used"])
    print("")
    print("wrote %s in %.1f seconds" % (out, report["seconds"]))
    return report


if __name__ == "__main__":
    main()
