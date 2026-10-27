"""Validate a fitted model on a feature table without refitting.

  python scripts/validate.py severson data/processed/features_sandia.csv     # cross-dataset
  python scripts/validate.py severson data/processed/features_severson.csv --holdout
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from fit_grading import clean
from oracle.grading import MODEL_DIR
from oracle.predict import load_model
from oracle.validate import holdout_report, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="name of models/grading_<model>.joblib")
    ap.add_argument("features")
    ap.add_argument("--holdout", action="store_true", help="fit on 2/3 of cells, test on the rest")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--own-reference", action="store_true",
                    help="use the table's own fresh median as reference (synthetic data)")
    args = ap.parse_args()

    df = clean(pd.read_csv(args.features))
    if args.holdout:
        mech_path = MODEL_DIR / f"mechanisms_{args.model}.joblib"
        mech = joblib.load(mech_path) if mech_path.exists() else None
        res = {f"seed_{s}": holdout_report(df, args.model, seed=s, mechanisms=mech)
               for s in range(args.seeds)}
    else:
        res = report(load_model(args.model), df, own_reference=args.own_reference)
    out = Path("data/processed") / f"validation_{args.model}_on_{Path(args.features).stem}{'_holdout' if args.holdout else ''}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    print("->", out)


if __name__ == "__main__":
    main()
