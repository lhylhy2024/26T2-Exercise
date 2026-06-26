"""
visualization.py — 数据可视化模块
功能：静态图表 (Matplotlib/Seaborn)、交互式图表 (Pyecharts)、词云图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from utils import setup_chinese_font, setup_plotting_style, ensure_dir

# 初始化绘图配置
setup_chinese_font()
setup_plotting_style()


def load_feature_data(path: str = "./data/feature/full_feature_data.csv") -> pd.DataFrame:
    """读取特征宽表"""
    print(f"  读取特征宽表: {path}")
    df = pd.read_csv(path)
    print(f"  加载完成: {len(df):,} 行, {len(df.columns)} 列")
    return df


# ==================== EDA 分析 ====================

def eda_basic_stats(df: pd.DataFrame) -> None:
    """基础数值统计分析"""
    print("\n" + "=" * 50)
    print("EDA：基础数值统计")
    print("=" * 50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().to_string())
    print()


def eda_category_heat(df: pd.DataFrame) -> pd.Series:
    """商品类目热度统计"""
    print("\n" + "=" * 50)
    print("EDA：商品类目热度")
    print("=" * 50)
    if "category_path" in df.columns:
        cate_count = df["category_path"].value_counts().head(15)
        for cat, cnt in cate_count.items():
            bar = "█" * int(cnt / cate_count.max() * 30)
            print(f"  {str(cat):>10s}: {cnt:4d}  {bar}")
        return cate_count
    else:
        print("  (无 category_path 字段)")
        return pd.Series(dtype=int)


def eda_hourly_trend(df: pd.DataFrame) -> pd.Series:
    """用户下单时段规律分析"""
    print("\n" + "=" * 50)
    print("EDA：下单时段分布")
    print("=" * 50)
    if "order_hour" in df.columns:
        hour_dist = df.groupby("order_hour").size()
        for h in range(24):
            cnt = hour_dist.get(h, 0)
            bar = "█" * int(cnt / hour_dist.max() * 25)
            print(f"  {h:02d}:00  {cnt:3d}  {bar}")
        return hour_dist
    else:
        print("  (无 order_hour 字段)")
        return pd.Series(dtype=int)


def eda_cross_analysis(df: pd.DataFrame) -> None:
    """类目营收交叉分析"""
    print("\n" + "=" * 50)
    print("EDA：类目营收交叉分析")
    print("=" * 50)
    if "category_path" in df.columns:
        cross = df.groupby("category_path").agg(
            订单数=("id_x", "count") if "id_x" in df.columns else ("id", "count"),
            平均价格=("price", "mean"),
            平均购买量=("quantity", "mean")
        ).sort_values("订单数", ascending=False).head(10)
        print(cross.to_string())
    else:
        print("  (无法执行交叉分析)")


def run_eda_analysis(df: pd.DataFrame) -> None:
    """
    执行完整 EDA 分析
    """
    eda_basic_stats(df)
    eda_category_heat(df)
    eda_hourly_trend(df)
    eda_cross_analysis(df)


# ==================== 静态可视化 ====================

def plot_category_heatmap(df: pd.DataFrame, top_n: int = 15,
                          save_path: str = "./img/category_heatmap.png") -> None:
    """
    绘制商品类目热度柱状图 (Matplotlib/Seaborn)
    """
    print("\n--- 绘制类目热度柱状图 ---")
    ensure_dir(os.path.dirname(save_path))

    if "category_path" not in df.columns:
        print("  (无 category_path 字段，跳过)")
        return

    cate_count = df["category_path"].value_counts().head(top_n)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=cate_count.values, y=cate_count.index, palette="viridis")
    ax.set_title("商品类目热度 TOP15", fontsize=16, fontweight="bold")
    ax.set_xlabel("订单数")
    ax.set_ylabel("类目路径")

    for i, v in enumerate(cate_count.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


def plot_hourly_trend(df: pd.DataFrame,
                      save_path: str = "./img/hourly_trend.png") -> None:
    """
    绘制下单时段趋势折线图
    """
    print("\n--- 绘制下单时段趋势图 ---")
    ensure_dir(os.path.dirname(save_path))

    if "order_hour" not in df.columns:
        print("  (无 order_hour 字段，跳过)")
        return

    hour_dist = df.groupby("order_hour").size()
    hours = list(range(24))
    counts = [hour_dist.get(h, 0) for h in hours]

    plt.figure(figsize=(14, 6))
    plt.plot(hours, counts, marker="o", linestyle="-", linewidth=2,
             color="#2E86AB", markersize=8)
    plt.xticks(range(0, 24, 2))
    plt.xlabel("小时")
    plt.ylabel("订单数")
    plt.title("全天各时段下单量趋势", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # 标注峰值
    max_hour = counts.index(max(counts))
    plt.annotate(f"峰值: {max_hour}:00 ({max(counts)}单)",
                 xy=(max_hour, max(counts)),
                 xytext=(max_hour + 2, max(counts) * 0.9),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=11, color="red")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ==================== 交互式可视化 ====================

def plot_interactive_category(df: pd.DataFrame, top_n: int = 15,
                              save_path: str = "./img/interactive_category.html") -> None:
    """
    绘制交互式类目柱状图 (Pyecharts)
    """
    print("\n--- 绘制交互式类目柱状图 ---")
    ensure_dir(os.path.dirname(save_path))

    try:
        from pyecharts.charts import Bar
        from pyecharts import options as opts
    except ImportError:
        print("  pyecharts 未安装，跳过交互式图表")
        return

    if "category_path" not in df.columns:
        print("  (无 category_path 字段，跳过)")
        return

    cate_count = df["category_path"].value_counts().head(top_n).sort_values()

    bar = (
        Bar()
        .add_xaxis([str(x) for x in cate_count.index.tolist()])
        .add_yaxis("订单数", cate_count.values.tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="商品类目热度 TOP15 (交互式)"),
            xaxis_opts=opts.AxisOpts(axislabel_opts={"rotate": 45}),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )
    bar.render(save_path)
    print(f"  已保存: {save_path}")


# ==================== 词云可视化 ====================

def plot_wordcloud(df: pd.DataFrame,
                   save_path: str = "./img/wordcloud.png") -> None:
    """
    生成商品标签词云图
    """
    print("\n--- 生成商品标签词云 ---")
    ensure_dir(os.path.dirname(save_path))

    try:
        from wordcloud import WordCloud
        import jieba
    except ImportError:
        print("  wordcloud 或 jieba 未安装，跳过词云")
        return

    # 合并所有标签文本
    tag_col = None
    for col in ["tags_prod", "tags"]:
        if col in df.columns:
            tag_col = col
            break

    if tag_col is None:
        print("  (无标签字段，跳过)")
        return

    # 分词
    all_tags = " ".join(df[tag_col].dropna().astype(str).tolist())
    words = " ".join(jieba.cut(all_tags))

    # 生成词云
    wc = WordCloud(
        font_path="C:/Windows/Fonts/simhei.ttf",
        width=1200,
        height=800,
        background_color="white",
        max_words=100,
        collocations=False
    ).generate(words)

    plt.figure(figsize=(16, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("商品标签词云图", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ==================== 统一入口 ====================

def run_all_visualizations(df: pd.DataFrame, output_dir: str = "./img") -> None:
    """
    执行所有可视化流程（EDA + 静态图 + 交互式 + 词云）
    """
    print("\n" + "=" * 60)
    print("第 3 阶段：可视化分析")
    print("=" * 60)

    # EDA
    run_eda_analysis(df)

    # 静态图
    plot_category_heatmap(df, save_path=os.path.join(output_dir, "category_heatmap.png"))
    plot_hourly_trend(df, save_path=os.path.join(output_dir, "hourly_trend.png"))

    # 交互式
    plot_interactive_category(df, save_path=os.path.join(output_dir, "interactive_category.html"))

    # 词云
    plot_wordcloud(df, save_path=os.path.join(output_dir, "wordcloud.png"))

    print("\n✅ 可视化分析完成！所有图表已保存至 img/ 目录")


if __name__ == "__main__":
    df = load_feature_data()
    run_all_visualizations(df)
