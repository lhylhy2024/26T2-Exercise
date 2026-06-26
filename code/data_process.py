"""
data_process.py — 数据读取、清洗、特征工程与多表融合
"""

import pandas as pd
import numpy as np
import os

from utils import read_big_csv, clean_log, ensure_dir


def load_raw_data(data_dir: str = "./data/raw") -> dict:
    """
    加载四份原始 CSV 数据

    Parameters
    ----------
    data_dir : str
        原始数据目录路径

    Returns
    -------
    dict
        {"users": df, "products": df, "browses": df, "orders": df}
    """
    print("=" * 50)
    print("阶段：加载原始数据")
    print("=" * 50)

    files = {
        "users": "users.csv",
        "products": "products.csv",
        "browses": "browses.csv",
        "orders": "orders.csv",
    }

    data = {}
    for name, fname in files.items():
        path = os.path.join(data_dir, fname)
        print(f"\n  读取 {fname} ...")
        if os.path.getsize(path) > 100 * 1024 * 1024:  # > 100MB 则分块
            data[name] = read_big_csv(path)
        else:
            data[name] = pd.read_csv(path)
            print(f"  读取完成: 共 {len(data[name]):,} 行")

    return data


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """清洗用户表"""
    print("\n--- 清洗 users ---")
    origin_len = len(df)

    # 1. 去重
    df = df.drop_duplicates(subset=['id'])

    # 2. 处理关键字段缺失
    df = df.dropna(subset=['gender', 'age'])

    # 3. 过滤异常年龄
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]

    clean_log(pd.DataFrame({"": [0] * origin_len}), pd.DataFrame({"": [0] * len(df)}), "users")
    # 手动输出日志
    removed = origin_len - len(df)
    print(f"  [users] {origin_len:,} → {len(df):,} 行 (删除 {removed:,} 行, {removed/origin_len*100:.2f}%)")
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """清洗商品表"""
    print("\n--- 清洗 products ---")
    origin_len = len(df)

    # 1. 去重
    df = df.drop_duplicates(subset=['id'])

    # 2. 处理异常价格（价格 <= 0 无意义）
    df = df[df['price'] > 0]

    # 3. 剔除标签为空的行
    df = df.dropna(subset=['name', 'tags'])

    removed = origin_len - len(df)
    print(f"  [products] {origin_len:,} → {len(df):,} 行 (删除 {removed:,} 行, {removed/origin_len*100:.2f}%)")
    return df


def clean_browses(df: pd.DataFrame) -> pd.DataFrame:
    """清洗浏览表"""
    print("\n--- 清洗 browses ---")
    origin_len = len(df)

    # 1. 删除 user_id 或 product_id 为空的行
    df = df.dropna(subset=['user_id', 'product_id'])

    # 2. 删除浏览时长为 0 或负数的记录
    df = df[df['duration'] >= 0]

    removed = origin_len - len(df)
    print(f"  [browses] {origin_len:,} → {len(df):,} 行 (删除 {removed:,} 行, {removed/origin_len*100:.2f}%)")
    return df


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """清洗订单表"""
    print("\n--- 清洗 orders ---")
    origin_len = len(df)

    # 1. 去重
    df = df.drop_duplicates(subset=['id'])

    # 2. 处理异常金额和数量
    df = df[df['total_amount'] >= 0]
    df = df[df['quantity'] > 0]

    # 3. 删除缺失关联键
    df = df.dropna(subset=['user_id', 'product_id'])

    removed = origin_len - len(df)
    print(f"  [orders] {origin_len:,} → {len(df):,} 行 (删除 {removed:,} 行, {removed/origin_len*100:.2f}%)")
    return df


def save_clean_data(data: dict, output_dir: str = "./data/clean") -> None:
    """保存清洗后数据"""
    ensure_dir(output_dir)
    print("\n--- 保存清洗数据 ---")
    for name, df in data.items():
        path = os.path.join(output_dir, f"{name}_clean.csv")
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"  已保存: {path} ({len(df):,} 行)")


def run_data_cleaning() -> dict:
    """
    执行完整数据清洗流程

    Returns
    -------
    dict
        清洗后的四份 DataFrame
    """
    print("=" * 50)
    print("第 2 阶段：数据清洗")
    print("=" * 50)

    # 加载
    raw = load_raw_data()

    # 分表清洗
    clean = {
        "users": clean_users(raw["users"]),
        "products": clean_products(raw["products"]),
        "browses": clean_browses(raw["browses"]),
        "orders": clean_orders(raw["orders"]),
    }

    # 保存
    save_clean_data(clean)

    return clean


def time_feature_engineering(df: pd.DataFrame, time_col: str = "created_at") -> pd.DataFrame:
    """
    对时间字段进行标准化与特征衍生

    Parameters
    ----------
    df : pd.DataFrame
        包含时间字段的数据
    time_col : str
        时间字段列名

    Returns
    -------
    pd.DataFrame
        添加了时间特征后的数据
    """
    # 转 datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # 衍生特征
    df["order_hour"] = df[time_col].dt.hour
    df["order_weekday"] = df[time_col].dt.weekday
    df["order_month"] = df[time_col].dt.month
    df["order_date"] = df[time_col].dt.date

    print(f"  时间特征衍生完成: hour/weekday/month")
    return df


def merge_tables(clean: dict) -> pd.DataFrame:
    """
    四表左关联融合，构建特征宽表

    Parameters
    ----------
    clean : dict
        清洗后的四份 DataFrame

    Returns
    -------
    pd.DataFrame
        融合后的特征宽表
    """
    print("\n" + "=" * 50)
    print("阶段：多表融合构建特征宽表")
    print("=" * 50)

    # 对 orders 做时间特征衍生
    orders = clean["orders"].copy()
    orders = time_feature_engineering(orders)

    # 左关联: orders ← users
    print("\n  关联 orders ← users (user_id)")
    merged = orders.merge(
        clean["users"][["id", "gender", "age", "city", "tags"]],
        left_on="user_id", right_on="id", how="left"
    )
    merged.drop(columns=["id_y"], inplace=True, errors="ignore")
    print(f"    当前维度: {merged.shape}")

    # 左关联: ← products
    print("  关联 ← products (product_id)")
    merged = merged.merge(
        clean["products"][["id", "name", "category_id", "category_path",
                           "price", "sales", "tags"]],
        left_on="product_id", right_on="id", how="left",
        suffixes=("_user", "_prod")
    )
    merged.drop(columns=["id"], inplace=True, errors="ignore")
    print(f"    当前维度: {merged.shape}")

    # 左关联: ← browses (聚合为浏览计数)
    print("  关联 ← browses (聚合浏览行为)")
    browse_agg = clean["browses"].groupby(["user_id", "product_id"]).agg(
        browse_count=("id", "count"),
        avg_duration=("duration", "mean")
    ).reset_index()
    merged = merged.merge(
        browse_agg,
        on=["user_id", "product_id"],
        how="left"
    )
    print(f"    最终维度: {merged.shape}")

    return merged


def save_feature_table(df: pd.DataFrame, output_dir: str = "./data/feature") -> None:
    """保存特征宽表"""
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "full_feature_data.csv")
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"\n  特征宽表已保存: {path} ({len(df):,} 行, {len(df.columns)} 列)")


def run_feature_engineering(clean: dict) -> pd.DataFrame:
    """
    执行完整特征工程与融合流程

    Returns
    -------
    pd.DataFrame
        特征宽表
    """
    feature_df = merge_tables(clean)
    save_feature_table(feature_df)
    return feature_df


def run_all() -> pd.DataFrame:
    """
    一键执行完整数据预处理流程（清洗 + 特征工程）

    Returns
    -------
    pd.DataFrame
        最终特征宽表
    """
    clean = run_data_cleaning()
    feature_df = run_feature_engineering(clean)
    return feature_df


if __name__ == "__main__":
    df = run_all()
    print(f"\n✅ 数据预处理完成！特征宽表: {df.shape}")
