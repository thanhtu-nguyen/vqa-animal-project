import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_columns(df):
    df = df.copy()

    required_cols = ["image", "question", "answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV thiếu cột bắt buộc: {col}")

    df["image"] = df["image"].astype(str).str.strip()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    return df


def prepare(train_csv, test_csv, image_dir, out_dir, val_ratio=0.1, seed=42):
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(train_csv)
    test_csv = Path(test_csv)

    print("[INFO] Đang chạy file prepare_data.py bản mới")
    print("[INFO] train_csv:", train_csv)
    print("[INFO] test_csv:", test_csv)
    print("[INFO] image_dir:", image_dir)
    print("[INFO] train folder:", image_dir / "train")
    print("[INFO] test folder:", image_dir / "test")
    print("[INFO] train folder exists:", (image_dir / "train").exists())
    print("[INFO] test folder exists:", (image_dir / "test").exists())

    train_df = pd.read_csv(train_csv, on_bad_lines="warn")
    test_df = pd.read_csv(test_csv, on_bad_lines="warn")

    train_df = normalize_columns(train_df)
    test_df = normalize_columns(test_df)

    train_df["split"] = "train"
    test_df["split"] = "test"

    # Ảnh nằm trực tiếp trong vqa_dataset/train và vqa_dataset/test
    train_df["image_path"] = train_df["image"].apply(
        lambda img: str(image_dir / "train" / img)
    )

    test_df["image_path"] = test_df["image"].apply(
        lambda img: str(image_dir / "test" / img)
    )

    # Debug 5 dòng đầu để chắc chắn đường dẫn đúng
    print("\n[DEBUG] 5 dòng train đầu:")
    print(train_df[["image", "image_path"]].head())

    print("\n[DEBUG] Kiểm tra tồn tại 5 ảnh train đầu:")
    for p in train_df["image_path"].head(5):
        print(p, "=>", Path(p).exists())

    train_df["image_exists"] = train_df["image_path"].apply(lambda p: Path(p).exists())
    test_df["image_exists"] = test_df["image_path"].apply(lambda p: Path(p).exists())

    missing_train = train_df[~train_df["image_exists"]].copy()
    missing_test = test_df[~test_df["image_exists"]].copy()

    if len(missing_train) > 0:
        print(f"\n[WARN] Train thiếu {len(missing_train)} dòng ảnh. Ví dụ:")
        print(missing_train[["image", "image_path", "question"]].head(10))
        missing_train.to_csv(
            out_dir / "missing_train_images.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if len(missing_test) > 0:
        print(f"\n[WARN] Test thiếu {len(missing_test)} dòng ảnh. Ví dụ:")
        print(missing_test[["image", "image_path", "question"]].head(10))
        missing_test.to_csv(
            out_dir / "missing_test_images.csv",
            index=False,
            encoding="utf-8-sig",
        )

    train_df = train_df[train_df["image_exists"]].reset_index(drop=True)
    test_df = test_df[test_df["image_exists"]].reset_index(drop=True)

    train_df = train_df.drop(columns=["image_exists"])
    test_df = test_df.drop(columns=["image_exists"])

    print(f"\n[INFO] Train hợp lệ: {len(train_df)} dòng")
    print(f"[INFO] Test hợp lệ: {len(test_df)} dòng")

    if len(train_df) == 0:
        raise ValueError(
            "Train rỗng sau khi kiểm tra ảnh. "
            "Hãy xem phần DEBUG ở trên để biết image_path đang sai ở đâu."
        )

    if len(test_df) == 0:
        print("[WARN] Test rỗng sau khi kiểm tra ảnh.")

    train_part, val_part = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_part.to_csv(out_dir / "train_prepared.csv", index=False, encoding="utf-8-sig")
    val_part.to_csv(out_dir / "val_prepared.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "test_prepared.csv", index=False, encoding="utf-8-sig")

    all_df = pd.concat([train_part, val_part, test_df], ignore_index=True)
    all_df.to_csv(out_dir / "all_prepared.csv", index=False, encoding="utf-8-sig")

    print("\n[DONE] Đã lưu dữ liệu vào:", out_dir)
    print(" - train_prepared.csv")
    print(" - val_prepared.csv")
    print(" - test_prepared.csv")
    print(" - all_prepared.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )