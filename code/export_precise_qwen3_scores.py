"""Export professor-ready Precise/Qwen3 comparison-score text files.

Each output line contains exactly:

    <image 1 filename> <image 2 filename> <matching score>

The primary genuine and primary impostor files are the two populations used
to compute the reported Precise similarity-scoring AUC. The separate
same-person/different-hand diagnostic comparisons are intentionally excluded.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAIR_FILE = PROJECT_ROOT / "results/precise/pairs_precise_eval.csv"
SCORE_FILE = (
    PROJECT_ROOT
    / "results/precise/qwen3vl/latest/"
    / "rb_qwen3vl_similarity_score_20260906_1552.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results/precise/qwen3vl"

CATEGORIES = {
    "primary_genuine": "res_precise_676_genuine_qwen3_vl-8b.txt",
    "primary_impostor": "res_precise_676_impostore_qwen3_vl-8b.txt",
}


def repair_missing_pair_ids(scores: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Recover blank IDs only when metadata maps to one unused manifest row."""
    known_ids = set(scores["pair_id"].dropna())
    match_columns = [
        "label",
        "category",
        "subject1",
        "subject2",
        "hand1",
        "hand2",
    ]

    for row_index in scores.index[scores["pair_id"].isna()]:
        row = scores.loc[row_index]
        candidates = pairs.copy()
        for column in match_columns:
            candidates = candidates[candidates[column] == row[column]]
        candidates = candidates[~candidates["pair_id"].isin(known_ids)]

        if len(candidates) != 1:
            raise ValueError(
                f"Cannot uniquely recover blank pair_id at score row {row_index}: "
                f"found {len(candidates)} candidates"
            )

        recovered_id = candidates.iloc[0]["pair_id"]
        scores.at[row_index, "pair_id"] = recovered_id
        known_ids.add(recovered_id)


def main() -> None:
    pairs = pd.read_csv(PAIR_FILE, dtype=str)
    scores = pd.read_csv(SCORE_FILE, dtype=str)
    repair_missing_pair_ids(scores, pairs)

    relevant_scores = scores[scores["category"].isin(CATEGORIES)].copy()
    if relevant_scores["pair_id"].duplicated().any():
        duplicates = relevant_scores.loc[
            relevant_scores["pair_id"].duplicated(keep=False), "pair_id"
        ].tolist()
        raise ValueError(f"Duplicate score pair IDs: {duplicates}")

    joined = pairs.merge(
        relevant_scores[["pair_id", "similarity_score"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for category, filename in CATEGORIES.items():
        category_rows = joined[joined["category"] == category].copy()
        if len(category_rows) != 676:
            raise ValueError(
                f"Expected 676 {category} comparisons, found {len(category_rows)}"
            )
        if category_rows["similarity_score"].isna().any():
            missing = category_rows.loc[
                category_rows["similarity_score"].isna(), "pair_id"
            ].tolist()
            raise ValueError(f"Missing scores for {category}: {missing}")

        output_path = OUTPUT_DIR / filename
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for row in category_rows.itertuples(index=False):
                score = float(row.similarity_score)
                formatted_score = str(int(score)) if score.is_integer() else str(score)
                output.write(
                    f"{Path(row.img1_path).name} "
                    f"{Path(row.img2_path).name} "
                    f"{formatted_score}\n"
                )

        print(f"{category}: {len(category_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
