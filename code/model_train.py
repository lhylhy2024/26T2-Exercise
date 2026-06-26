"""
model_train.py — 数据建模与预测模块
功能：RFM 用户分层、商品热度挖掘与销量预测、协同过滤推荐
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from utils import ensure_dir


# ==================== 1. RFM 用户价值分层 ====================

def rfm_segmentation(orders_path: str = "./data/raw/orders.csv",
                     output_dir: str = "./model_output") -> pd.DataFrame:
    """
    基于订单数据的 RFM 用户分层 + KMeans 聚类

    Parameters
    ----------
    orders_path : str
        订单数据路径
    output_dir : str
        输出目录

    Returns
    -------
    pd.DataFrame
        包含 RFM 指标和聚类标签的用户分层结果
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    print("\n" + "=" * 60)
    print("RFM 用户价值分层建模")
    print("=" * 60)

    # 1. 读取订单数据
    print("\n[1] 加载订单数据...")
    df = pd.read_csv(orders_path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    print(f"  订单总数: {len(df):,}")
    print(f"  时间范围: {df['created_at'].min()} ~ {df['created_at'].max()}")

    # 2. 计算 RFM 指标
    print("\n[2] 计算 RFM 指标...")
    max_date = df["created_at"].max()

    rfm = df.groupby("user_id").agg(
        R=("created_at", lambda x: (max_date - x.max()).days),
        F=("id", "count"),
        M=("quantity", "sum")
    ).reset_index()

    print(f"  有效用户数: {len(rfm):,}")
    print(f"\n  RFM 统计:")
    print(f"    R (距今天数): 均值={rfm['R'].mean():.1f}, 中位={rfm['R'].median():.1f}, "
          f"最大={rfm['R'].max()}")
    print(f"    F (订单数):   均值={rfm['F'].mean():.1f}, 中位={rfm['F'].median():.1f}, "
          f"最大={rfm['F'].max()}")
    print(f"    M (购买件数): 均值={rfm['M'].mean():.1f}, 中位={rfm['M'].median():.1f}, "
          f"最大={rfm['M'].max()}")

    # 3. 标准化 + KMeans 聚类
    print("\n[3] KMeans 聚类 (n_clusters=4)...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["R", "F", "M"]])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # 4. 输出各聚类均值
    print("\n[4] 各聚类均值:")
    cluster_stats = rfm.groupby("cluster")[["R", "F", "M"]].mean().round(1)
    cluster_counts = rfm["cluster"].value_counts().sort_index()
    cluster_stats["人数"] = cluster_counts
    cluster_stats["占比"] = (cluster_counts / len(rfm) * 100).round(1).astype(str) + "%"
    print(cluster_stats.to_string())

    # 5. 用户分层命名
    print("\n[5] 用户分层:")
    # 按 R 值排序映射: R 小(活跃) → 高价值, R 大(沉默) → 流失
    cluster_order = rfm.groupby("cluster")["R"].mean().sort_values().index.tolist()
    label_map = {
        cluster_order[0]: "高价值用户",
        cluster_order[1]: "潜力用户",
        cluster_order[2]: "沉睡用户",
        cluster_order[3]: "流失用户"
    }
    rfm["user_segment"] = rfm["cluster"].map(label_map)
    seg_dist = rfm["user_segment"].value_counts()
    for seg, cnt in seg_dist.items():
        print(f"    {seg}: {cnt:,} 人 ({cnt/len(rfm)*100:.1f}%)")

    # 6. 保存
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "rfm_segmentation.csv")
    rfm.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n  已保存: {out_path}")

    return rfm


# ==================== 2. 商品热度挖掘与销量预测 ====================

def sales_prediction(feature_path: str = "./data/feature/full_feature_data.csv",
                     output_dir: str = "./model_output") -> dict:
    """
    商品热点评分 + 随机森林销量预测

    Parameters
    ----------
    feature_path : str
        特征宽表路径
    output_dir : str
        输出目录

    Returns
    -------
    dict
        包含热度榜单和模型评估结果
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    print("\n" + "=" * 60)
    print("商品热度挖掘与销量预测")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载特征宽表...")
    df = pd.read_csv(feature_path)
    print(f"  数据: {df.shape}")

    # 2. 商品热度计算
    print("\n[2] 计算商品综合热度...")
    product_heat = df.groupby("product_id").agg(
        销量=("quantity", "sum"),
        订单数=("id_x", "count") if "id_x" in df.columns else ("id", "count"),
        平均价格=("price", "mean"),
        平均浏览时长=("avg_duration", "mean"),
    ).reset_index()

    # 综合热度评分 (加权)
    product_heat["热度评分"] = (
        product_heat["销量"] * 0.3 +
        product_heat["订单数"] * 0.4 +
        product_heat["平均浏览时长"].fillna(0) * 0.1 +
        (1 / (product_heat["平均价格"] + 1)) * 0.2
    )
    product_heat.sort_values("热度评分", ascending=False, inplace=True)

    print("\n  热度 TOP10:")
    top10 = product_heat.head(10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {i:2d}. 商品 {int(row['product_id'])} | "
              f"销量={int(row['销量'])} | 订单={int(row['订单数'])} | "
              f"热度={row['热度评分']:.2f}")

    # 3. 随机森林销量预测
    print("\n[3] 随机森林销量预测...")
    # 构造特征 (使用 10% 样本训练，保证速度)
    model_df = df.dropna(subset=["price", "quantity"]).sample(frac=0.1, random_state=42)
    features = ["price"]
    if "order_hour" in model_df.columns:
        features.append("order_hour")
    if "age" in model_df.columns:
        features.append("age")
        model_df["age"] = model_df["age"].fillna(model_df["age"].median())
    if "gender" in model_df.columns:
        features.append("gender")
        model_df["gender"] = model_df["gender"].fillna(0)

    X = model_df[features]
    y = model_df["quantity"]

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 预测与评估
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n  模型评估:")
    print(f"    R² = {r2:.4f}")
    print(f"    MAE = {mae:.2f} (平均绝对误差)")
    print(f"    MSE = {mse:.2f} (均方误差)")
    print(f"    特征重要性: {dict(zip(features, rf.feature_importances_.round(4)))}")

    # 4. 保存结果
    ensure_dir(output_dir)
    product_heat.to_csv(os.path.join(output_dir, "product_heat_ranking.csv"),
                        index=False, encoding='utf-8')
    print(f"\n  热度榜单已保存")

    return {
        "heat_top10": top10,
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "feature_importance": dict(zip(features, rf.feature_importances_.round(4)))
    }


# ==================== 3. 协同过滤推荐 ====================

def collaborative_recommend(orders_path: str = "./data/raw/orders.csv",
                            top_n: int = 5,
                            target_user_id: int = None,
                            output_dir: str = "./model_output") -> dict:
    """
    基于用户协同过滤的个性化推荐

    Parameters
    ----------
    orders_path : str
        订单数据路径
    top_n : int
        推荐商品数量
    target_user_id : int, optional
        指定推荐用户，None 则使用第一个用户

    Returns
    -------
    dict
        推荐结果
    """
    print("\n" + "=" * 60)
    print("协同过滤个性化推荐")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 构建用户-商品交互矩阵 (稀疏矩阵)...")
    df = pd.read_csv(orders_path)

    # 只取最近活跃的 N 个用户 (控制内存)
    top_users = df["user_id"].value_counts().head(2000).index
    df_sub = df[df["user_id"].isin(top_users)]

    # 构建用户-商品交互计数
    interaction = df_sub.groupby(["user_id", "product_id"]).size().reset_index(name="count")

    # 使用 scipy 稀疏矩阵
    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity

    user_ids = interaction["user_id"].unique()
    product_ids = interaction["product_id"].unique()
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    prod_map = {pid: j for j, pid in enumerate(product_ids)}

    rows = interaction["user_id"].map(user_map)
    cols = interaction["product_id"].map(prod_map)
    data = interaction["count"]

    sparse_matrix = csr_matrix((data, (rows, cols)),
                                shape=(len(user_ids), len(product_ids)))
    print(f"  稀疏矩阵维度: {sparse_matrix.shape} ({len(user_ids)} 用户 × {len(product_ids)} 商品)")
    print(f"  非零元素: {sparse_matrix.nnz:,}")
    print(f"  矩阵密度: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]) * 100:.4f}%")

    # 2. 计算用户相似度 (使用稀疏矩阵避免内存爆炸)
    print("\n[2] 计算用户相似度...")
    user_sim = cosine_similarity(sparse_matrix, dense_output=False)
    print(f"  相似度矩阵已构建")

    # 3. 推荐函数 (基于稀疏矩阵直接查询)
    def recommend(user_id, n=top_n):
        if user_id not in user_map:
            return f"用户 {user_id} 不在活跃用户列表中"

        uid_idx = user_map[user_id]
        # 获取最相似的 5 个用户 (排除自身)
        sim_row = user_sim[uid_idx].toarray().flatten()
        sim_row[uid_idx] = -1  # 排除自身
        top_sim_indices = sim_row.argsort()[-6:-1][::-1]
        top_sim_users = [user_ids[i] for i in top_sim_indices]

        # 获取目标用户已购买的商品
        bought = set(df[df["user_id"] == user_id]["product_id"].unique())

        # 从相似用户的购买中推荐
        candidates = df[df["user_id"].isin(top_sim_users)]
        rec_scores = candidates.groupby("product_id").size()
        rec_scores = rec_scores[~rec_scores.index.isin(bought)]
        recommendations = rec_scores.sort_values(ascending=False).head(n)

        return recommendations

    # 4. 执行推荐
    if target_user_id is None:
        target_user_id = user_ids[0]

    print(f"\n[3] 为用户 {target_user_id} 推荐商品...")
    result = recommend(target_user_id, top_n)

    print(f"\n  推荐结果 (TOP {top_n}):")
    if isinstance(result, str):
        print(f"  {result}")
    else:
        for i, (pid, score) in enumerate(result.items(), 1):
            print(f"  {i:2d}. 商品 {pid}  (推荐得分: {score:.0f})")

    # 5. 保存
    ensure_dir(output_dir)
    if not isinstance(result, str):
        rec_df = pd.DataFrame({
            "product_id": result.index,
            "recommend_score": result.values
        })
        rec_df.to_csv(os.path.join(output_dir, f"recommend_user_{target_user_id}.csv"),
                      index=False, encoding='utf-8')
        print(f"\n  推荐结果已保存")

    return {
        "user_id": target_user_id,
        "recommendations": result,
        "matrix_shape": (len(user_ids), len(product_ids))
    }


# ==================== 统一入口 ====================

def run_all_models() -> dict:
    """
    执行全部建模流程

    Returns
    -------
    dict
        各模型结果
    """
    print("\n" + "=" * 60)
    print("第 4 阶段：数据建模与预测")
    print("=" * 60)

    results = {}

    print("\n" + "-" * 40)
    results["rfm"] = rfm_segmentation()

    print("\n" + "-" * 40)
    results["sales_pred"] = sales_prediction()

    print("\n" + "-" * 40)
    results["recommend"] = collaborative_recommend()

    print("\n✅ 全部建模完成！")
    return results


if __name__ == "__main__":
    run_all_models()
