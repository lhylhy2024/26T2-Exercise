"""
main.py — 一键主程序
功能：按阶段依次调用各模块，实现全流程自动化
"""

import sys
import os
import time

# 确保可以导入同级目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_chinese_font, setup_plotting_style, ensure_dir
from data_process import run_data_cleaning, run_feature_engineering
from visualization import (
    load_feature_data, run_all_visualizations
)
from model_train import rfm_segmentation, sales_prediction, collaborative_recommend


def print_header(text: str) -> None:
    """打印阶段标题"""
    width = 70
    print("\n" + "#" * width)
    print(f"## {text:^62} ##")
    print("#" * width)


def print_step(step_num: int, total: int, text: str) -> None:
    """打印步骤进度"""
    print(f"\n>> 步骤 {step_num}/{total}: {text}")
    print("-" * 50)


def run_pipeline() -> None:
    """
    执行全流程数据分析与挖掘管线

    阶段说明:
      第 1 阶段: 环境搭建与数据准备 (已在 Notebook 中完成)
      第 2 阶段: 数据预处理 (清洗 + 特征工程)
      第 3 阶段: 数据可视化分析 (EDA + 静态/交互式图表 + 词云)
      第 4 阶段: 数据建模与预测 (RFM + 销量预测 + 推荐模型)
      第 5 阶段: 项目总结
    """
    total_steps = 5
    overall_start = time.time()

    print("\n" + "★" * 70)
    print("  Python 电商大数据分析与商品推荐挖掘 — 全流程自动执行")
    print("★" * 70)
    print(f"\n  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  工作目录: {os.getcwd()}")

    # ============================
    # 初始化
    # ============================
    print_header("初始化")
    try:
        print("[*] 配置中文字体与绘图样式...")
        setup_chinese_font()
        setup_plotting_style()
        ensure_dir("./model_output")
        ensure_dir("./img")
        print("[OK] 初始化完成")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        return

    # ============================
    # 第 2 阶段: 数据预处理
    # ============================
    print_header("第 2 阶段 — 数据预处理")
    step_start = time.time()
    try:
        print_step(1, total_steps, "数据清洗")
        clean_data = run_data_cleaning()

        print_step(2, total_steps, "特征工程与多表融合")
        feature_df = run_feature_engineering(clean_data)

        elapsed = time.time() - step_start
        print(f"\n[OK] 数据预处理完成，耗时 {elapsed:.1f} 秒")
        print(f"     特征宽表: {feature_df.shape}")

    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        print("     请确保原始数据已放置在 data/raw 目录中")
        return
    except Exception as e:
        print(f"[ERROR] 数据预处理失败: {e}")
        return

    # ============================
    # 第 3 阶段: 数据可视化分析
    # ============================
    print_header("第 3 阶段 — 数据可视化分析")
    step_start = time.time()
    try:
        print_step(3, total_steps, "EDA 分析与可视化呈现")

        # 如果第 2 阶段未执行，尝试从文件加载
        if "feature_df" not in dir() or feature_df is None:
            try:
                feature_df = load_feature_data()
            except FileNotFoundError:
                print("[ERROR] 特征宽表不存在，请先执行数据预处理")
                return

        run_all_visualizations(feature_df)

        elapsed = time.time() - step_start
        print(f"\n[OK] 可视化分析完成，耗时 {elapsed:.1f} 秒")

    except ImportError as e:
        print(f"[WARN] 可视化库导入失败: {e}")
        print("     跳过可视化阶段，继续执行建模...")
    except Exception as e:
        print(f"[ERROR] 可视化分析失败: {e}")
        print("     跳过可视化阶段，继续执行建模...")

    # ============================
    # 第 4 阶段: 数据建模与预测
    # ============================
    print_header("第 4 阶段 — 数据建模与预测")
    step_start = time.time()
    try:
        print_step(4, total_steps, "RFM 用户分层")
        try:
            rfm_result = rfm_segmentation()
        except Exception as e:
            print(f"[WARN] RFM 建模失败: {e}")

        print_step(5, total_steps, "销量预测与协同过滤推荐")
        try:
            sales_result = sales_prediction()
        except Exception as e:
            print(f"[WARN] 销量预测失败: {e}")

        try:
            rec_result = collaborative_recommend()
        except Exception as e:
            print(f"[WARN] 推荐模型失败: {e}")

        elapsed = time.time() - step_start
        print(f"\n[OK] 建模完成，耗时 {elapsed:.1f} 秒")

    except Exception as e:
        print(f"[ERROR] 建模阶段失败: {e}")

    # ============================
    # 完成
    # ============================
    total_elapsed = time.time() - overall_start
    print_header("全部完成")
    print(f"\n  结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"\n  输出目录:")
    print(f"    - 清洗数据:  ./data/clean/")
    print(f"    - 特征宽表:  ./data/feature/")
    print(f"    - 可视化图表: ./img/")
    print(f"    - 模型结果:   ./model_output/")
    print("\n" + "★" * 70)
    print("  🎉 全流程执行完毕！")
    print("★" * 70)


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n[!] 用户中断执行")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] 未预期错误: {e}")
        sys.exit(1)
