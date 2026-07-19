"""
setup.py
--------
Automated setup script to run the entire data & ML pipeline in one command.
Executes:
    1. database.py           — Download Olist dataset & build SQLite DB
    2. preprocessing.py      — Perform data cleaning and validation
    3. feature_engineering.py — Compute 15 customer behavioral features
    4. train.py              — Train XGBoost pipeline & save artifacts
"""
import subprocess
import sys
import time


def run_script(name: str) -> None:
    print(f"\n==================================================")
    print(f"🚀 Running: {name} ...")
    print(f"==================================================")
    start_time = time.time()
    try:
        res = subprocess.run([sys.executable, name], check=True)
        duration = time.time() - start_time
        print(f"✅ Finished {name} in {duration:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while running {name}: {e}")
        sys.exit(e.returncode)


def main():
    start_all = time.time()
    scripts = [
        "database.py",
        "preprocessing.py",
        "feature_engineering.py",
        "train.py",
    ]
    for script in scripts:
        run_script(script)

    total_time = time.time() - start_all
    print(f"\n==================================================")
    print(f"🎉 Pipeline setup complete! Total time: {total_time:.1f}s")
    print(f"👉 To launch the dashboard, run: streamlit run app.py")
    print(f"==================================================\n")


if __name__ == "__main__":
    main()
