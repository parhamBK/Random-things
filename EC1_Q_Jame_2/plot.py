import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    })


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0 or np.isnan(sigma):
        return np.zeros_like(x, dtype=float)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def suggested_bins(values: np.ndarray, min_bins: int = 6, max_bins: int = 18) -> int:
    """Freedman–Diaconis rule with sane limits."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return min_bins

    q25, q75 = np.percentile(v, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return min_bins

    width = 2 * iqr / (v.size ** (1 / 3))
    if width <= 0:
        return min_bins

    bins = int(math.ceil((v.max() - v.min()) / width))
    return int(np.clip(bins, min_bins, max_bins))


def render_table_page(df: pd.DataFrame, title: str, page: int, pages: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    if pages > 1:
        ax.set_title(f"{title} (Page {page}/{pages})", pad=16, fontweight="bold")
    else:
        ax.set_title(title, pad=16, fontweight="bold")


    show = df.copy()

    for col in show.columns:
        if col.lower().endswith("score") or col.lower() in {"mean", "variance", "std"}:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
        else:
            show[col] = show[col].astype(str)

    table = ax.table(
        cellText=show.values,
        colLabels=list(show.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

    fig.tight_layout()
    return fig


def paginate_table(df: pd.DataFrame, title: str, rows_per_page: int) -> list[plt.Figure]:
    total_rows = len(df)
    total_pages = max(1, int(math.ceil(total_rows / rows_per_page))) if total_rows else 1

    figs: list[plt.Figure] = []
    for i in range(total_pages):
        start = i * rows_per_page
        end = min((i + 1) * rows_per_page, total_rows)
        chunk = df.iloc[start:end]
        figs.append(render_table_page(chunk, title, page=i + 1, pages=total_pages))

    return figs


def render_histogram(scores: pd.Series, title: str, mu: float, sigma: float) -> plt.Figure:
    vals = pd.to_numeric(scores, errors="coerce").dropna().astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 5))
    if vals.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        fig.tight_layout()
        return fig

    bins = suggested_bins(vals)
    ax.hist(vals, bins=bins, density=True, alpha=0.85, edgecolor="white", linewidth=1.0)

    x = np.linspace(max(0, vals.min() - 5), min(100, vals.max() + 5), 500)
    ax.plot(x, normal_pdf(x, mu, sigma), linewidth=2.2)
    ax.axvline(mu, linestyle="--", linewidth=1.8)

    ax.set_xlim(0, 100)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")

    ax.text(
        0.98, 0.95,
        f"n={vals.size}\nμ={mu:.2f}\nσ={sigma:.2f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.9),
    )

    fig.tight_layout()
    return fig
