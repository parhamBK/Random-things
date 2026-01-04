import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from stats import Stats, describe
from plot import paginate_table, render_histogram


def rescale_quiz2_to_quiz1(df: pd.DataFrame, quiz1_stats: Stats, quiz2_stats: Stats) -> pd.Series:
    """Z-score Quiz 2 using its own stats, then map it to Quiz 1 mean/std."""
    if quiz2_stats.std == 0 or np.isnan(quiz2_stats.std):
        raise ValueError("Quiz 2 std is 0 (or NaN); cannot rescale.")

    z = (df["score"] - quiz2_stats.mean) / quiz2_stats.std
    scaled = z * quiz1_stats.std + quiz1_stats.mean
    return scaled.clip(0, 100)


def build_report_pdf(df: pd.DataFrame, out_path: str = "quiz_report.pdf") -> None:
    quizzes = sorted(df["quiz"].dropna().unique())
    stats_by_quiz = {q: describe(df.loc[df["quiz"] == q, "score"]) for q in quizzes}

    summary = pd.DataFrame({
        "Quiz": quizzes,
        "N": [stats_by_quiz[q].n for q in quizzes],
        "Mean": [stats_by_quiz[q].mean for q in quizzes],
        "Variance": [stats_by_quiz[q].var for q in quizzes],
        "Std": [stats_by_quiz[q].std for q in quizzes],
    })

    df = df.copy()
    df["final_score"] = df["score"]

    if "Quiz 1" in stats_by_quiz and "Quiz 2" in stats_by_quiz:
        q1 = stats_by_quiz["Quiz 1"]
        q2 = stats_by_quiz["Quiz 2"]
        mask = df["quiz"] == "Quiz 2"
        df.loc[mask, "final_score"] = rescale_quiz2_to_quiz1(df.loc[mask], q1, q2)

    final_scores = (
        df[["student", "quiz", "final_score"]]
        .sort_values(["quiz", "student"])
        .reset_index(drop=True)
    )

    figures: list[plt.Figure] = []
    figures += paginate_table(summary, "Summary Statistics", rows_per_page=20)
    figures += paginate_table(final_scores, "Student Final Scores", rows_per_page=30)

    for q in quizzes:
        st = stats_by_quiz[q]
        figures.append(render_histogram(
            df.loc[df["quiz"] == q, "score"],
            f"{q} — Score Distribution",
            st.mean,
            st.std,
        ))

    if "Quiz 1" in stats_by_quiz and "Quiz 2" in stats_by_quiz:
        q1 = stats_by_quiz["Quiz 1"]
        figures.append(render_histogram(
            df.loc[df["quiz"] == "Quiz 2", "final_score"],
            "Quiz 2 — Final Scores",
            q1.mean,
            q1.std,
        ))

    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ {out_path} created successfully.")
