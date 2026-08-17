from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DETAIL_PATH = (
    BASE_DIR
    / "reports"
    / "answer_quality_details.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "reports"
    / "retriever_final_analysis.txt"
)

IMPROVED_PATH = (
    BASE_DIR
    / "reports"
    / "v2_best_cases.csv"
)

DEGRADED_PATH = (
    BASE_DIR
    / "reports"
    / "v2_worst_cases.csv"
)


# ------------------------------------------------------------
# 이 정도 차이 이하는 사실상 동일하다고 판단
# ------------------------------------------------------------

EPSILON = 1e-4


def main():

    print("=" * 70)
    print("FINAL RETRIEVER ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        DETAIL_PATH
    )

    print(
        f"Samples: {len(df):,}"
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    df["delta"] = (
        df["v2_answer_similarity_best3"]
        - df["pretrained_answer_similarity_best3"]
    )

    # --------------------------------------------------------
    # mutually exclusive Win / Lose / Tie
    # --------------------------------------------------------

    v2_win_mask = (
        df["delta"]
        > EPSILON
    )

    pretrained_win_mask = (
        df["delta"]
        < -EPSILON
    )

    tie_mask = (
        df["delta"]
        .abs()
        <= EPSILON
    )

    v2_wins = int(
        v2_win_mask.sum()
    )

    pretrained_wins = int(
        pretrained_win_mask.sum()
    )

    ties = int(
        tie_mask.sum()
    )

    total = len(
        df
    )

    v2_win_rate = (
        v2_wins / total
    )

    pretrained_win_rate = (
        pretrained_wins / total
    )

    tie_rate = (
        ties / total
    )

    # --------------------------------------------------------
    # Non-tie comparison
    # --------------------------------------------------------

    non_tie_count = (
        v2_wins
        + pretrained_wins
    )

    if non_tie_count > 0:

        v2_non_tie_win_rate = (
            v2_wins
            / non_tie_count
        )

    else:

        v2_non_tie_win_rate = 0.0

    # --------------------------------------------------------
    # Delta distribution
    # --------------------------------------------------------

    mean_delta = float(
        df["delta"].mean()
    )

    median_delta = float(
        df["delta"].median()
    )

    p10 = float(
        df["delta"].quantile(
            0.10
        )
    )

    p90 = float(
        df["delta"].quantile(
            0.90
        )
    )

    # --------------------------------------------------------
    # Top1 retrieved ticket 자체가 같은지
    # --------------------------------------------------------

    same_top1 = (
        df["pretrained_subject"]
        .fillna("")
        .astype(str)
        ==
        df["v2_subject"]
        .fillna("")
        .astype(str)
    )

    same_top1_rate = float(
        same_top1.mean()
    )

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    improved_df = (
        df.sort_values(
            "delta",
            ascending=False,
        )
        .head(50)
        .copy()
    )

    degraded_df = (
        df.sort_values(
            "delta",
            ascending=True,
        )
        .head(50)
        .copy()
    )

    improved_df.to_csv(
        IMPROVED_PATH,
        index=False,
    )

    degraded_df.to_csv(
        DEGRADED_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    result = f"""
FINAL RETRIEVER ANALYSIS
======================================================================

Samples:
{total}

Tie Threshold:
{EPSILON}


WIN / LOSE / TIE
----------------------------------------------------------------------

V2 Wins:
{v2_wins}
Rate:
{v2_win_rate:.4f}

Pretrained Wins:
{pretrained_wins}
Rate:
{pretrained_win_rate:.4f}

Tie:
{ties}
Rate:
{tie_rate:.4f}

Total:
{v2_win_rate + pretrained_win_rate + tie_rate:.4f}


NON-TIE COMPARISON
----------------------------------------------------------------------

Different Results:
{non_tie_count}

V2 Win Rate among non-ties:
{v2_non_tie_win_rate:.4f}


ANSWER SIMILARITY DELTA
----------------------------------------------------------------------

Mean Delta:
{mean_delta:+.6f}

Median Delta:
{median_delta:+.6f}

10th Percentile:
{p10:+.6f}

90th Percentile:
{p90:+.6f}


RETRIEVAL BEHAVIOR
----------------------------------------------------------------------

Same Top1 Ticket Rate:
{same_top1_rate:.4f}
"""

    print(
        result
    )

    OUTPUT_PATH.write_text(
        result,
        encoding="utf-8",
    )

    print()
    print(
        "Best V2 cases:"
    )
    print(
        IMPROVED_PATH
    )

    print()
    print(
        "Worst V2 cases:"
    )
    print(
        DEGRADED_PATH
    )


if __name__ == "__main__":
    main()