"""
utils.py — 通用工具函数模块
功能：分块读取、清洗日志、中文绘图配置等可复用工具
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免阻塞
import matplotlib.pyplot as plt
import os


def read_big_csv(file_path: str, chunk_size: int = 100000, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    分块读取大文件 CSV，避免内存溢出

    Parameters
    ----------
    file_path : str
        CSV 文件路径
    chunk_size : int
        每块读取行数，默认 10000
    encoding : str
        文件编码，默认 utf-8

    Returns
    -------
    pd.DataFrame
        合并后的完整 DataFrame
    """
    chunks = []
    reader = pd.read_csv(file_path, chunksize=chunk_size, encoding=encoding)
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        print(f"  已读取第 {i+1} 块 ({len(chunk)} 行)")
    df = pd.concat(chunks, ignore_index=True)
    print(f"  读取完成: 共 {len(df):,} 行")
    return df


def clean_log(origin_df: pd.DataFrame, clean_df: pd.DataFrame, name: str) -> None:
    """
    输出数据清洗前后的对比日志

    Parameters
    ----------
    origin_df : pd.DataFrame
        清洗前原始数据
    clean_df : pd.DataFrame
        清洗后数据
    name : str
        数据表名称（如 "users"）
    """
    removed = len(origin_df) - len(clean_df)
    ratio = removed / len(origin_df) * 100 if len(origin_df) > 0 else 0
    print(f"  [{name}] {len(origin_df):,} → {len(clean_df):,} 行 "
          f"(删除 {removed:,} 行, {ratio:.2f}%)")


def setup_chinese_font() -> None:
    """
    配置 Matplotlib 中文字体（SimHei），解决中文乱码
    """
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("  [OK] 中文字体已配置 (SimHei)")


def setup_plotting_style() -> None:
    """
    设置全局绘图样式
    """
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12
    print("  [OK] 绘图样式已配置")


def ensure_dir(dir_path: str) -> None:
    """
    确保目录存在，若不存在则创建

    Parameters
    ----------
    dir_path : str
        目录路径
    """
    os.makedirs(dir_path, exist_ok=True)
