import pandas as pd

from data import DATA
from plot import configure_matplotlib
from report import build_report_pdf


def main() -> None:
    configure_matplotlib()

    df = pd.DataFrame(DATA)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    build_report_pdf(df, out_path="quiz_report.pdf")


if __name__ == "__main__":
    main()
