"""automate_Muhammad_Aldi_Alfatih.py

Script otomasi preprocessing dataset Breast Cancer Wisconsin.
Menjalankan seluruh pipeline preprocessing dari data mentah hingga
menghasilkan file train.csv dan test.csv yang siap untuk pelatihan model.

Cara menjalankan:
    python automate_Muhammad_Aldi_Alfatih.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Konfigurasi ──────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parents[1] / "breast_cancer_raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "breast_cancer_preprocessing"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "target"


def load_raw_dataset() -> pd.DataFrame:
    """Memuat dataset Breast Cancer Wisconsin dari sklearn dan menyimpan versi mentahnya."""
    print("[1/5] Memuat dataset mentah dari sklearn...")
    data = load_breast_cancer()
    df_raw = pd.DataFrame(data.data, columns=data.feature_names)
    df_raw[TARGET_COLUMN] = data.target

    # Simpan data mentah jika belum ada
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "breast_cancer_raw.csv"
    if not raw_path.exists():
        df_raw.to_csv(raw_path, index=False)
        print(f"   ✓ Data mentah disimpan ke: {raw_path}")
    else:
        print(f"   ✓ Data mentah sudah ada: {raw_path}")

    print(f"   Dataset shape: {df_raw.shape}")
    return df_raw


def explore_data(df: pd.DataFrame) -> None:
    """Menampilkan informasi dasar dataset."""
    print("\n[2/5] Eksplorasi data...")
    print(f"   Jumlah baris   : {len(df)}")
    print(f"   Jumlah kolom   : {len(df.columns)}")
    print(f"   Missing values : {df.isnull().sum().sum()}")
    print(f"   Distribusi target:")
    print(f"     - Malignant (0): {(df[TARGET_COLUMN] == 0).sum()}")
    print(f"     - Benign    (1): {(df[TARGET_COLUMN] == 1).sum()}")


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Melakukan preprocessing: feature scaling dan train-test split."""
    print("\n[3/5] Preprocessing...")

    # Pisahkan fitur dan target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train size : {len(X_train)} sampel")
    print(f"   Test size  : {len(X_test)} sampel")

    # StandardScaler — fit pada train, transform pada keduanya
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # Gabungkan kembali dengan target
    train_df = X_train_scaled.copy()
    train_df[TARGET_COLUMN] = y_train.values

    test_df = X_test_scaled.copy()
    test_df[TARGET_COLUMN] = y_test.values

    print("   ✓ StandardScaler diterapkan (fit pada train saja)")
    return train_df, test_df


def save_results(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Menyimpan hasil preprocessing ke folder output."""
    print("\n[4/5] Menyimpan hasil preprocessing...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"   ✓ train.csv disimpan  → {train_path} ({len(train_df)} baris)")
    print(f"   ✓ test.csv disimpan   → {test_path} ({len(test_df)} baris)")


def main() -> None:
    print("=" * 60)
    print("  AUTOMATE PREPROCESSING - Muhammad Aldi Alfatih")
    print("  Dataset: Breast Cancer Wisconsin")
    print("=" * 60)

    df_raw = load_raw_dataset()
    explore_data(df_raw)
    train_df, test_df = preprocess(df_raw)
    save_results(train_df, test_df)

    print("\n[5/5] Selesai!")
    print("=" * 60)
    print("  Output tersimpan di folder: preprocessing/breast_cancer_preprocessing/")
    print("  Siap untuk digunakan pada tahap Membangun_model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
