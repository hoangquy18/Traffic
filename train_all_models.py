#!/usr/bin/env python3
"""Script to train all models (ML and DL) with and without weather features.

Chạy grid search/scan cho:
- ML models: Decision Tree, XGBoost
- DL models: RNN, GNN, Transformer, TCN, Informer, TimesNet, GMAN++

Với 2 modes:
1. WITH-WEATHER: Sử dụng tất cả features (traffic + weather)
2. NO-WEATHER: Chỉ sử dụng traffic features

Kết quả sẽ được lưu vào:
- ML: experiments/ml_scan_with_weather.csv và experiments/ml_scan_no_weather.csv
- DL: experiments/dl_scan_with_weather.csv và experiments/dl_scan_no_weather.csv
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_command(cmd: list, description: str, cwd: Path = None) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}\n")
    else:
        print()

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    project_root = Path(__file__).parent.absolute()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            cwd=cwd or project_root,
            env=env,
        )
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train all models (ML and DL) with and without weather features"
    )
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Only run ML models (Decision Tree, XGBoost)",
    )
    parser.add_argument(
        "--dl-only",
        action="store_true",
        help="Only run DL models (RNN, GNN, Transformer, etc.)",
    )
    parser.add_argument(
        "--with-weather-only",
        action="store_true",
        help="Only run with weather features",
    )
    parser.add_argument(
        "--no-weather-only",
        action="store_true",
        help="Only run without weather features",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=1,
        help="Minimum sequence_length (default: 1)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=6,
        help="Maximum sequence_length (default: 6)",
    )
    parser.add_argument(
        "--min-horizon",
        type=int,
        default=1,
        help="Minimum prediction_horizon (default: 1)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=5,
        help="Maximum prediction_horizon (default: 5)",
    )
    args = parser.parse_args()

    # Determine what to run
    run_ml = not args.dl_only
    run_dl = not args.ml_only
    run_with_weather = not args.no_weather_only
    run_no_weather = not args.with_weather_only

    if not (run_ml or run_dl):
        print("❌ Error: Must run at least ML or DL models")
        sys.exit(1)

    if not (run_with_weather or run_no_weather):
        print("❌ Error: Must run at least with-weather or no-weather mode")
        sys.exit(1)

    # Get script directory (project root)
    script_dir = Path(__file__).parent.absolute()
    ml_script = script_dir / "traffic_trainer" / "trainers" / "ml_grid_search.py"
    dl_script = script_dir / "traffic_trainer" / "trainers" / "dl_scan.py"

    if not ml_script.exists():
        print(f"❌ Error: ML script not found at {ml_script}")
        sys.exit(1)
    if not dl_script.exists():
        print(f"❌ Error: DL script not found at {dl_script}")
        sys.exit(1)

    print(f"📁 Project root: {script_dir}")
    print(f"📄 ML script: {ml_script}")
    print(f"📄 DL script: {dl_script}\n")

    results = []

    # ============================================
    # ML Models
    # ============================================
    if run_ml:
        # ML with weather
        if run_with_weather:
            cmd = [
                sys.executable,
                str(ml_script),
                "--results-csv",
                "experiments/ml_scan_with_weather.csv",
            ]
            success = run_command(
                cmd, "ML Models - WITH WEATHER (Decision Tree, XGBoost)", cwd=script_dir
            )
            results.append(("ML - WITH WEATHER", success))

        # ML without weather
        if run_no_weather:
            cmd = [
                sys.executable,
                str(ml_script),
                "--no-weather",
                "--results-csv",
                "experiments/ml_scan_no_weather.csv",
            ]
            success = run_command(
                cmd, "ML Models - NO WEATHER (Decision Tree, XGBoost)", cwd=script_dir
            )
            results.append(("ML - NO WEATHER", success))

    # ============================================
    # DL Models
    # ============================================
    if run_dl:
        dl_base_cmd = [
            sys.executable,
            str(dl_script),
            "--min-seq-len",
            str(args.min_seq_len),
            "--max-seq-len",
            str(args.max_seq_len),
            "--min-horizon",
            str(args.min_horizon),
            "--max-horizon",
            str(args.max_horizon),
        ]

        # DL with weather
        if run_with_weather:
            cmd = dl_base_cmd + [
                "--results-csv",
                "experiments/dl_scan_with_weather.csv",
            ]
            success = run_command(
                cmd,
                "DL Models - WITH WEATHER (RNN, GNN, Transformer, TCN, TimesNet, GMAN++)",
                cwd=script_dir,
            )
            results.append(("DL - WITH WEATHER", success))

        # DL without weather
        if run_no_weather:
            cmd = dl_base_cmd + [
                "--no-weather",
                "--results-csv",
                "experiments/dl_scan_no_weather.csv",
            ]
            success = run_command(
                cmd,
                "DL Models - NO WEATHER (RNN, GNN, Transformer, TCN, TimesNet, GMAN++)",
                cwd=script_dir,
            )
            results.append(("DL - NO WEATHER", success))

    # ============================================
    # Summary
    # ============================================
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}\n")

    all_success = True
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {name}")
        if not success:
            all_success = False

    print(f"\n{'='*80}")
    if all_success:
        print("🎉 All training completed successfully!")
        print("\nResults saved to:")
        if run_ml:
            if run_with_weather:
                print("  - experiments/ml_scan_with_weather.csv")
            if run_no_weather:
                print("  - experiments/ml_scan_no_weather.csv")
        if run_dl:
            if run_with_weather:
                print("  - experiments/dl_scan_with_weather.csv")
            if run_no_weather:
                print("  - experiments/dl_scan_no_weather.csv")
    else:
        print("⚠️  Some training runs failed. Check output above for details.")
        sys.exit(1)

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
