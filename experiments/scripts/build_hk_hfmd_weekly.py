import pandas as pd
from pathlib import Path


def main() -> None:
    base_dir = Path("experiments/data_for_model/手足口病/data_HK")
    out_path = base_dir / "hk_hfmd_weekly_2010_2025.csv"

    all_dfs = []
    for year in range(2010, 2026):
        csv_path = base_dir / f"ha_hfm_{year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Parse week end date (format like 02/01/2010)
        df["week_end_date"] = pd.to_datetime(df["week_end_date"], format="%d/%m/%Y")

        # Total admissions (with + without complication)
        df["admissions_total"] = df["adm_without_complication"] + df["adm_with_complication"]
        df["year"] = year
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values("week_end_date")

    # ISO year/week metadata
    iso = full["week_end_date"].dt.isocalendar()
    full["iso_year"] = iso.year
    full["iso_week"] = iso.week

    full = full[
        [
            "week_end_date",
            "year",
            "iso_year",
            "iso_week",
            "adm_without_complication",
            "adm_with_complication",
            "admissions_total",
        ]
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    print(f"Saved merged HK HFMD weekly series to: {out_path}")


if __name__ == "__main__":
    main()
