"""Tổng hợp kết quả scan từ ml_scan.csv và dl_scan.csv theo từng model."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_and_merge_csvs(ml_scan_path: Path, dl_scan_path: Path) -> pd.DataFrame:
    """Load và merge 2 file CSV, chuẩn hóa cột metric."""
    # Đọc với engine='python' để xử lý các giá trị có dấu phẩy (như tuple)
    try:
        ml_df = pd.read_csv(ml_scan_path, engine="python", on_bad_lines="skip")
    except TypeError:
        # pandas < 1.3
        try:
            ml_df = pd.read_csv(
                ml_scan_path,
                engine="python",
                error_bad_lines=False,
                warn_bad_lines=False,
            )
        except Exception:
            ml_df = pd.read_csv(ml_scan_path, engine="python", sep=",", quotechar='"')

    try:
        dl_df = pd.read_csv(dl_scan_path, engine="python", on_bad_lines="skip")
    except TypeError:
        # pandas < 1.3
        try:
            dl_df = pd.read_csv(
                dl_scan_path,
                engine="python",
                error_bad_lines=False,
                warn_bad_lines=False,
            )
        except Exception:
            dl_df = pd.read_csv(dl_scan_path, engine="python", sep=",", quotechar='"')

    # Chuẩn hóa tên cột metric
    # ml_scan có: score_f1_macro, val_f1_macro
    # dl_scan có: best_val_f1, val_f1_macro
    # -> dùng val_f1_macro làm primary metric, fallback sang best_val_f1 hoặc score_f1_macro

    if "val_f1_macro" in ml_df.columns:
        ml_df["f1_macro"] = ml_df["val_f1_macro"]
    elif "score_f1_macro" in ml_df.columns:
        ml_df["f1_macro"] = ml_df["score_f1_macro"]
    else:
        ml_df["f1_macro"] = 0.0

    if "val_f1_macro" in dl_df.columns:
        dl_df["f1_macro"] = dl_df["val_f1_macro"]
    elif "best_val_f1" in dl_df.columns:
        dl_df["f1_macro"] = dl_df["best_val_f1"]
    else:
        dl_df["f1_macro"] = 0.0

    # Merge
    merged = pd.concat([ml_df, dl_df], ignore_index=True)

    return merged


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp theo từng model."""
    summary_rows = []

    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()

        # Overall best
        if model_df["f1_macro"].notna().any():
            best_idx = model_df["f1_macro"].idxmax()
            if pd.notna(best_idx):
                best_row = model_df.loc[best_idx]
            else:
                continue
        else:
            continue

        summary_rows.append(
            {
                "model": model,
                "metric": "overall_best",
                "f1_macro": best_row["f1_macro"],
                "sequence_length": best_row.get("sequence_length", ""),
                "prediction_horizon": best_row.get("prediction_horizon", ""),
                "val_accuracy": best_row.get("val_accuracy", ""),
                "val_f1_weighted": best_row.get("val_f1_weighted", ""),
                "val_precision_macro": best_row.get("val_precision_macro", ""),
                "val_recall_macro": best_row.get("val_recall_macro", ""),
                "output_dir": best_row.get("output_dir", ""),
            }
        )

        # Best per sequence_length
        if "sequence_length" in model_df.columns:
            for seq_len in sorted(model_df["sequence_length"].unique()):
                seq_df = model_df[model_df["sequence_length"] == seq_len]
                if len(seq_df) > 0 and seq_df["f1_macro"].notna().any():
                    best_seq_idx = seq_df["f1_macro"].idxmax()
                    if pd.notna(best_seq_idx):
                        best_seq_row = seq_df.loc[best_seq_idx]
                    else:
                        continue
                else:
                    continue
                    summary_rows.append(
                        {
                            "model": model,
                            "metric": f"best_seq_len_{seq_len}",
                            "f1_macro": best_seq_row["f1_macro"],
                            "sequence_length": seq_len,
                            "prediction_horizon": best_seq_row.get(
                                "prediction_horizon", ""
                            ),
                            "val_accuracy": best_seq_row.get("val_accuracy", ""),
                            "val_f1_weighted": best_seq_row.get("val_f1_weighted", ""),
                            "val_precision_macro": best_seq_row.get(
                                "val_precision_macro", ""
                            ),
                            "val_recall_macro": best_seq_row.get(
                                "val_recall_macro", ""
                            ),
                            "output_dir": best_seq_row.get("output_dir", ""),
                        }
                    )

        # Best per prediction_horizon
        if "prediction_horizon" in model_df.columns:
            for horizon in sorted(model_df["prediction_horizon"].unique()):
                h_df = model_df[model_df["prediction_horizon"] == horizon]
                if len(h_df) > 0 and h_df["f1_macro"].notna().any():
                    best_h_idx = h_df["f1_macro"].idxmax()
                    if pd.notna(best_h_idx):
                        best_h_row = h_df.loc[best_h_idx]
                        summary_rows.append(
                            {
                                "model": model,
                                "metric": f"best_horizon_{horizon}",
                                "f1_macro": best_h_row["f1_macro"],
                                "sequence_length": best_h_row.get(
                                    "sequence_length", ""
                                ),
                                "prediction_horizon": horizon,
                                "val_accuracy": best_h_row.get("val_accuracy", ""),
                                "val_f1_weighted": best_h_row.get(
                                    "val_f1_weighted", ""
                                ),
                                "val_precision_macro": best_h_row.get(
                                    "val_precision_macro", ""
                                ),
                                "val_recall_macro": best_h_row.get(
                                    "val_recall_macro", ""
                                ),
                                "output_dir": best_h_row.get("output_dir", ""),
                            }
                        )

        # Stats
        summary_rows.append(
            {
                "model": model,
                "metric": "mean_f1",
                "f1_macro": model_df["f1_macro"].mean(),
                "sequence_length": "",
                "prediction_horizon": "",
                "val_accuracy": "",
                "val_f1_weighted": "",
                "val_precision_macro": "",
                "val_recall_macro": "",
                "output_dir": "",
            }
        )

        summary_rows.append(
            {
                "model": model,
                "metric": "std_f1",
                "f1_macro": model_df["f1_macro"].std(),
                "sequence_length": "",
                "prediction_horizon": "",
                "val_accuracy": "",
                "val_f1_weighted": "",
                "val_precision_macro": "",
                "val_recall_macro": "",
                "output_dir": "",
            }
        )

    return pd.DataFrame(summary_rows)


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng so sánh các model."""
    comparison_rows = []

    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()

        comparison_rows.append(
            {
                "model": model,
                "best_f1_macro": model_df["f1_macro"].max(),
                "mean_f1_macro": model_df["f1_macro"].mean(),
                "std_f1_macro": model_df["f1_macro"].std(),
                "min_f1_macro": model_df["f1_macro"].min(),
                "num_runs": len(model_df),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("best_f1_macro", ascending=False)

    return comparison_df


def create_by_timestep_horizon_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng tổng hợp theo từng combination của sequence_length và prediction_horizon."""
    if "sequence_length" not in df.columns or "prediction_horizon" not in df.columns:
        return pd.DataFrame()

    # Lọc bỏ các dòng không có đủ thông tin
    df_clean = df[
        df["sequence_length"].notna() & df["prediction_horizon"].notna()
    ].copy()

    if len(df_clean) == 0:
        return pd.DataFrame()

    timestep_horizon_rows = []

    # Lấy tất cả combinations của sequence_length và prediction_horizon
    seq_lengths = sorted(df_clean["sequence_length"].unique())
    horizons = sorted(df_clean["prediction_horizon"].unique())

    for seq_len in seq_lengths:
        for horizon in horizons:
            # Lọc data cho combination này
            combo_df = df_clean[
                (df_clean["sequence_length"] == seq_len)
                & (df_clean["prediction_horizon"] == horizon)
            ].copy()

            if len(combo_df) == 0:
                continue

            # Tìm best model cho combination này
            if combo_df["f1_macro"].notna().any():
                best_idx = combo_df["f1_macro"].idxmax()
                best_row = combo_df.loc[best_idx]

                # Lấy precision/recall từ best model, nếu không có thì lấy từ model có precision/recall tốt nhất
                precision = best_row.get("val_precision_macro", "")
                recall = best_row.get("val_recall_macro", "")

                # Nếu best model không có precision/recall, tìm model có precision/recall tốt nhất
                if (
                    pd.isna(precision) or precision == ""
                ) and "val_precision_macro" in combo_df.columns:
                    precision_df = combo_df[combo_df["val_precision_macro"].notna()]
                    if len(precision_df) > 0:
                        best_precision_idx = precision_df[
                            "val_precision_macro"
                        ].idxmax()
                        precision = precision_df.loc[best_precision_idx][
                            "val_precision_macro"
                        ]

                if (
                    pd.isna(recall) or recall == ""
                ) and "val_recall_macro" in combo_df.columns:
                    recall_df = combo_df[combo_df["val_recall_macro"].notna()]
                    if len(recall_df) > 0:
                        best_recall_idx = recall_df["val_recall_macro"].idxmax()
                        recall = recall_df.loc[best_recall_idx]["val_recall_macro"]

                timestep_horizon_rows.append(
                    {
                        "sequence_length": int(seq_len),
                        "prediction_horizon": int(horizon),
                        "best_model": best_row["model"],
                        "best_f1_macro": best_row["f1_macro"],
                        "best_val_accuracy": best_row.get("val_accuracy", ""),
                        "best_val_f1_weighted": best_row.get("val_f1_weighted", ""),
                        "best_val_precision_macro": (
                            precision if precision != "" else ""
                        ),
                        "best_val_recall_macro": recall if recall != "" else "",
                        "num_models_tested": len(combo_df["model"].unique()),
                        "all_models": ", ".join(sorted(combo_df["model"].unique())),
                    }
                )

    result_df = pd.DataFrame(timestep_horizon_rows)
    if len(result_df) > 0:
        result_df = result_df.sort_values(["sequence_length", "prediction_horizon"])

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tổng hợp kết quả scan từ ml_scan.csv và dl_scan.csv"
    )
    parser.add_argument(
        "--ml-scan",
        type=Path,
        default=Path("ml_scan.csv"),
        help="Đường dẫn file ml_scan.csv",
    )
    parser.add_argument(
        "--dl-scan",
        type=Path,
        default=Path("dl_scan.csv"),
        help="Đường dẫn file dl_scan.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("scan_summary.csv"),
        help="File CSV output cho summary chi tiết",
    )
    parser.add_argument(
        "--output-comparison",
        type=Path,
        default=Path("scan_comparison.csv"),
        help="File CSV output cho bảng so sánh",
    )
    parser.add_argument(
        "--output-timestep-horizon",
        type=Path,
        default=Path("scan_by_timestep_horizon.csv"),
        help="File CSV output cho bảng theo sequence_length và prediction_horizon",
    )
    args = parser.parse_args()

    print(f"Loading {args.ml_scan}...")
    print(f"Loading {args.dl_scan}...")

    merged_df = load_and_merge_csvs(args.ml_scan, args.dl_scan)
    print(f"\nTotal rows: {len(merged_df)}")
    print(f"Models: {sorted(merged_df['model'].unique())}")

    # Tổng hợp theo model
    print("\nGenerating summary...")
    summary_df = summarize_by_model(merged_df)
    summary_df.to_csv(args.output_summary, index=False)
    print(f"✅ Summary saved to: {args.output_summary}")

    # Bảng so sánh
    print("\nGenerating comparison table...")
    comparison_df = create_comparison_table(merged_df)
    comparison_df.to_csv(args.output_comparison, index=False)
    print(f"✅ Comparison saved to: {args.output_comparison}")

    # Bảng theo timestep và horizon
    print("\nGenerating timestep-horizon table...")
    timestep_horizon_df = create_by_timestep_horizon_table(merged_df)
    if len(timestep_horizon_df) > 0:
        timestep_horizon_df.to_csv(args.output_timestep_horizon, index=False)
        print(f"✅ Timestep-Horizon table saved to: {args.output_timestep_horizon}")
    else:
        print("⚠ No timestep-horizon data available")

    # Print top models
    print("\n" + "=" * 80)
    print("TOP MODELS (by best F1 macro):")
    print("=" * 80)
    print(comparison_df.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL (best configurations):")
    print("=" * 80)
    for model in sorted(merged_df["model"].unique()):
        model_summary = summary_df[summary_df["model"] == model]
        overall_best = model_summary[model_summary["metric"] == "overall_best"]
        if len(overall_best) > 0:
            row = overall_best.iloc[0]
            print(
                f"\n{model:20s} | Best F1: {row['f1_macro']:.4f} | "
                f"Seq: {row['sequence_length']} | Horizon: {row['prediction_horizon']}"
            )

    # Print by timestep and horizon
    if len(timestep_horizon_df) > 0:
        print("\n" + "=" * 80)
        print("BEST MODEL BY SEQUENCE LENGTH AND PREDICTION HORIZON:")
        print("=" * 80)
        print(timestep_horizon_df.to_string(index=False))

        # Print dạng pivot table để dễ đọc hơn
        print("\n" + "=" * 80)
        print(
            "PIVOT TABLE: Best F1 Macro by Sequence Length (rows) and Horizon (cols):"
        )
        print("=" * 80)
        pivot_df = timestep_horizon_df.pivot(
            index="sequence_length",
            columns="prediction_horizon",
            values="best_f1_macro",
        )
        print(pivot_df.to_string())

        print("\n" + "=" * 80)
        print("PIVOT TABLE: Best Model by Sequence Length (rows) and Horizon (cols):")
        print("=" * 80)
        pivot_model_df = timestep_horizon_df.pivot(
            index="sequence_length", columns="prediction_horizon", values="best_model"
        )
        print(pivot_model_df.to_string())

        # Pivot table cho Precision
        if "best_val_precision_macro" in timestep_horizon_df.columns:
            print("\n" + "=" * 80)
            print(
                "PIVOT TABLE: Best Precision Macro by Sequence Length (rows) and Horizon (cols):"
            )
            print("=" * 80)
            pivot_precision_df = timestep_horizon_df.pivot(
                index="sequence_length",
                columns="prediction_horizon",
                values="best_val_precision_macro",
            )
            print(pivot_precision_df.to_string())

        # Pivot table cho Recall
        if "best_val_recall_macro" in timestep_horizon_df.columns:
            print("\n" + "=" * 80)
            print(
                "PIVOT TABLE: Best Recall Macro by Sequence Length (rows) and Horizon (cols):"
            )
            print("=" * 80)
            pivot_recall_df = timestep_horizon_df.pivot(
                index="sequence_length",
                columns="prediction_horizon",
                values="best_val_recall_macro",
            )
            print(pivot_recall_df.to_string())


if __name__ == "__main__":
    main()
