from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# Reuse the same ordered keyword mapping as preprocess (abridged to match core diseases)
DISEASE_KEYWORDS_ORDERED: List[Tuple[str, List[str]]] = [
    ("甲型肝炎", ["甲型肝炎", "甲肝"]),
    ("乙型肝炎", ["乙型肝炎", "乙肝"]),
    ("丙型肝炎", ["丙型肝炎", "丙肝"]),
    ("丁型肝炎", ["丁型肝炎", "丁肝"]),
    ("戊型肝炎", ["戊型肝炎", "戊肝"]),
    ("病毒性肝炎", ["病毒性肝炎", "肝炎"]),
    ("人禽流感", ["人禽流感"]),
    ("人感染H7N9禽流感", ["H7N9"]),
    ("流行性感冒", ["流行性感冒", "流感"]),
    ("新型冠状病毒肺炎", ["新型冠状病毒肺炎", "冠状病毒", "新冠"]),
    ("传染性非典", ["传染性非典", "非典", "SARS"]),
    ("细菌性和阿米巴性痢疾", ["细菌性和阿米巴性痢疾", "痢疾"]),
    ("其他感染性腹泻病", ["其他感染性腹泻病", "腹泻病"]),
    ("艾滋病", ["艾滋病", "AIDS"]),
    ("百日咳", ["百日咳"]),
    ("流行性腮腺炎", ["流行性腮腺炎", "腮腺炎"]),
    ("手足口病", ["手足口病", "手足口"]),
    ("肺结核", ["肺结核", "结核"]),
]


def map_diagnosis(diagnosis: str) -> str:
    if pd.isna(diagnosis):
        return "Other"
    s = str(diagnosis).strip().replace(" ", "")
    for name, kws in DISEASE_KEYWORDS_ORDERED:
        for k in kws:
            if k in s:
                return name
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--lis_path", default=str(Path(__file__).resolve().parents[2] / "data" / "LIS 去除身份证.xlsx"))
    args = ap.parse_args()

    week0 = datetime.strptime(args.week_start, "%Y-%m-%d").date()
    week1 = week0 + timedelta(days=max(1, args.days))

    df = pd.read_excel(args.lis_path)
    if "INSPECTION_DATE" not in df.columns:
        raise SystemExit("LIS 파일에 INSPECTION_DATE 컬럼이 없습니다.")
    # parse date: supports both yyyymmdd and ISO
    # Robust date parse: try %Y%m%d first, then general parser
    ts = pd.to_datetime(df["INSPECTION_DATE"], format="%Y%m%d", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df["INSPECTION_DATE"], errors="coerce")
    df = df.assign(__date=ts.dt.date).dropna(subset=["__date"]).copy()
    one_week = df[(df["__date"] >= week0) & (df["__date"] < week1)].copy()

    # raw diagnosis diversity
    raw_col = "CLINICAL_DIAGNOSES" if "CLINICAL_DIAGNOSES" in one_week.columns else None
    if raw_col is None:
        print("[WARN] CLINICAL_DIAGNOSES 컬럼이 없어 원문 진단 다양성은 생략합니다.")
    else:
        one_week[raw_col] = one_week[raw_col].astype(str)
        n_raw_unique = one_week[raw_col].nunique()
        print(f"원문 진단 고유 개수: {n_raw_unique}")
        print("상위 30개 원문 진단:")
        print(one_week[raw_col].value_counts().head(30))

    # mapped epi_category
    if raw_col is not None:
        one_week["epi_category"] = one_week[raw_col].apply(map_diagnosis)
        print("\n매핑 후 진단 분포:")
        print(one_week["epi_category"].value_counts())
        # show flu-like variants in raw text
        flu_like = one_week[one_week[raw_col].str.contains("流感", na=False)][raw_col]
        if not flu_like.empty:
            print("\n'流感' 포함 원문 진단 예시(상위 30):")
            print(flu_like.value_counts().head(30))
    else:
        print("매핑 생략: CLINICAL_DIAGNOSES 없음")


if __name__ == "__main__":
    main()


