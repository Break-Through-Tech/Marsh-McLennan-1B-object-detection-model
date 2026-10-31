"""Build a readable report of how a prediction file scored against the gold labels.

Usage:
    ./.venv/bin/python src/make_report.py                      # spaCy baseline
    ./.venv/bin/python src/make_report.py --predictions data/processed/predictions_llm.csv --name llm

Reads data/gold/gold_sample.csv and the predictions file, writes
reports/<name>_report.md. It does not change either input.
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def singular(word: str) -> str:
    """Same crude singularization the scorer uses."""
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def parse_objects(value) -> set:
    if pd.isna(value):
        return set()
    items = {singular(p.strip().lower()) for p in str(value).split(";")}
    return {i for i in items if i and i != "none"}


def last_word_match(gold: set, pred: set) -> tuple:
    """Lenient score: 'carpeted floor' and 'floor' count as a match (same last word)."""
    g = {x.split()[-1] for x in gold}
    p = {x.split()[-1] for x in pred}
    return len(g & p), len(p - g), len(g - p)


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "data/gold/gold_sample.csv")
    ap.add_argument("--predictions", type=Path, default=ROOT / "data/processed/predictions_spacy.csv")
    ap.add_argument("--name", default="spacy_baseline")
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    gold = gold[gold["labeler"].notna() & (gold["labeler"].astype(str).str.strip() != "")]
    pred = pd.read_csv(args.predictions)
    df = gold.merge(pred, on="case_id", how="left", suffixes=("", "_pred"))
    df["gold_set"] = df["gold_objects"].apply(parse_objects)
    df["pred_set"] = df["objects"].apply(parse_objects)

    tp = fp = fn = 0
    ltp = lfp = lfn = 0
    fp_counter, fn_counter = Counter(), Counter()
    exact = 0
    wording_only = 0
    none_but_guessed = 0
    none_rows = 0
    wrong_rows = []
    for _, r in df.iterrows():
        g, p = r["gold_set"], r["pred_set"]
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        a, b, c = last_word_match(g, p)
        ltp += a; lfp += b; lfn += c
        fp_counter.update(p - g); fn_counter.update(g - p)
        if g == p:
            exact += 1
        else:
            wrong_rows.append(r)
            if last_word_match(g, p)[1] == 0 and last_word_match(g, p)[2] == 0:
                wording_only += 1
        if not g:
            none_rows += 1
            if p:
                none_but_guessed += 1

    n = len(df)
    pr, rc, f1 = scores(tp, fp, fn)
    lpr, lrc, lf1 = scores(ltp, lfp, lfn)

    lines = [f"# Report: {args.name}", "",
             f"Gold file: `{args.gold.name}` ({n} labeled narratives). Predictions: `{args.predictions.name}`.", "",
             "## Scores", "",
             md_table([
                 ["Exact string match (official)", f"{pr:.3f}", f"{rc:.3f}", f"{f1:.3f}"],
                 ["Last-word match (lenient)", f"{lpr:.3f}", f"{lrc:.3f}", f"{lf1:.3f}"],
             ], ["Scoring", "Precision", "Recall", "F1"]), "",
             f"- Exact-match accuracy (whole row right): **{exact}/{n} = {exact / n:.3f}**",
             f"- Correct objects (TP): {tp}, wrong guesses (FP): {fp}, missed objects (FN): {fn}",
             f"- Rows that are wrong only because of wording (e.g. gold `carpeted floor`, predicted `floor`): **{wording_only}**",
             f"- Rows labeled NONE: {none_rows}; the model still guessed objects for **{none_but_guessed}** of them", "",
             "The lenient score counts two objects as the same when their last word matches. "
             "A big gap between the two scores means the label wording, not the model, is costing points.", "",
             "## Most common wrong guesses (false positives)", "",
             md_table(fp_counter.most_common(15), ["Predicted object", "Times"]), "",
             "## Most common misses (false negatives)", "",
             md_table(fn_counter.most_common(15), ["Gold object", "Times"]), "",
             f"## {args.examples} example errors", ""]
    ex = []
    for r in wrong_rows[: args.examples]:
        ex.append([r["case_id"], str(r["incident_text"])[:110],
                   "; ".join(sorted(r["gold_set"])) or "NONE",
                   "; ".join(sorted(r["pred_set"])) or "(nothing)"])
    lines.append(md_table(ex, ["Case", "Narrative", "Gold", "Predicted"]))

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.name}_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print(f"precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}  exact match {exact / n:.3f}")


if __name__ == "__main__":
    main()
