# 26T2-Exercise
Second Term of ZQU-Exercise file archive.

# Python电商大数据分析与商品推荐挖掘实践指导书

**适用课程**：Python大数据分析、电商数据挖掘、智能推荐实训

**实训模式**：基于ShopRec-CN官方四源数据集，完成从环境搭建、数据处理、可视化分析到推荐模型落地的全流程实战

**实训核心依托**：本实践指导书基于ShopRec-CN官方原生四文件数据集（users.csv、browses.csv、products.csv、orders.csv），所有操作、代码、分析逻辑、建模流程均贴合真实电商业务场景。

**实训技术链路**：实训环境搭建→标准化项目架构搭建→多源数据集获取与读取→大数据分表清洗降噪→时间特征衍生与多表数据融合→EDA探索性商业分析→静态/交互式/词云多维可视化→RFM用户价值分层建模→商品热度挖掘与销量预测→协同过滤个性化推荐模型→项目工程化优化→成果汇总与答辩交付

## 通用实训规范

1. **版本规范**：全程使用Python3.9稳定版本，兼容Python3.10-3.14高版本，依托Anaconda统一环境管理，禁止单独安装Python，规避环境冲突报错；
2. **路径规范**：所有项目文件、文件夹命名纯英文、无中文、无空格、无特殊符号，杜绝路径读取异常；
3. **数据规范**：原始四源数据集统一归档至data/raw目录，仅只读使用、禁止直接修改，清洗、特征、建模数据分层独立存储；
4. **代码规范**：所有实训代码统一在Jupyter Notebook中分段编写、逐块调试，代码附带详细注释，阶段性成果全部截图留存；
5. **设备规范**：实训设备需保证8G及以上内存、10G以上剩余硬盘空间，运行大数据量代码时关闭代理，避免加载超时、内存溢出问题。

---

# 第一阶段 环境搭建和数据准备

## 模块一 实训环境搭建与大数据项目架构拆解

### 一、实训目的

1. 掌握Anaconda环境安装、虚拟环境创建、镜像源配置全套实操流程，适配大数据实训需求；
2. 熟练安装实训所需全部第三方库，解决下载超时、安装失败问题；
3. 手动+代码双方式搭建企业级标准化项目目录，掌握项目文件分层管理规范；
4. 完成环境校验，修复Python可视化中文乱码问题，确保后续所有代码可正常运行。

### 二、实训环境与资源准备

1. **硬件环境**：内存≥8G，硬盘剩余空间≥10G，支持大文件批量读写与模型运算；
2. **软件资源（国内极速镜像）**：
   - Anaconda清华镜像地址：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
   - 推荐安装版本：Anaconda3-2023.07（内置Python3.9，适配全部实训依赖与代码）
   - VS Code编辑器地址：https://code.visualstudio.com/
3. **核心注意事项**：仅需安装Anaconda软件，无需单独下载Python，避免双环境冲突导致库导入失败、内核异常等问题。

### 三、分步实操步骤

#### 步骤1：安装Anaconda3-2023.07版本
1. 打开上方清华镜像地址，下拉找到 **Anaconda3-2023.07-2-Windows-x86_64.exe** 版本并下载；
2. 双击安装包，安装路径全程无中文、无空格，建议默认路径；
3. 安装选项全部默认，无需手动勾选环境变量，安装完成后重启电脑。

#### 步骤2：打开终端并创建专属虚拟环境
1. 电脑开始菜单搜索并打开 **Anaconda Prompt**；
2. 输入创建环境命令，回车执行，输入y确认安装：
   ```bash
   conda create -n ecommerce_bigdata python=3.9 -y
   ```
3. 环境创建完成后，输入激活命令：
   ```bash
   conda activate ecommerce_bigdata
   ```
4. 看到前缀出现 `(ecommerce_bigdata)` 即为激活成功。

#### 步骤3：配置清华镜像源（解决下载慢、报错问题）
依次复制以下4条命令，逐条回车执行：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --remove-key defaults
```

#### 步骤4：一键安装全部实训依赖库
复制以下完整命令，回车批量安装所有所需库，等待全部安装完成无报错：
```bash
pip install pandas numpy matplotlib seaborn requests beautifulsoup4 pyecharts jieba scikit-learn modelscope tqdm wordcloud -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 步骤5：代码自动创建标准化项目目录
1. 终端输入 `jupyter notebook` 回车，打开网页编辑器；
2. 新建Python3笔记本，复制以下代码运行，自动生成所有项目文件夹：

```python
# 导入库
import os

# 定义项目目录列表
dirs = [
    "code",
    "data/raw",
    "data/clean",
    "data/feature",
    "img",
    "model_output",
    "report"
]

# 循环创建目录
for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"目录创建成功：{dir_path}")

print("✅ 全部项目目录创建完成！")
```

#### 步骤6：全局环境校验与中文乱码修复
运行以下测试代码，验证环境正常、中文显示正常：

```python
# 全局配置解决matplotlib中文乱码
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 导入核心库测试
import pandas as pd
import numpy as np
import seaborn as sns
import pyecharts
import sklearn

print("✅ Pandas版本：", pd.__version__)
print("✅ Numpy版本：", np.__version__)
print("✅ 所有环境配置成功，可正常开展实训！")
```

### 四、实训成果与截图要求
1. 截图终端虚拟环境激活成功界面（带 `(ecommerce_bigdata)` 前缀）；
2. 截图目录自动创建成功的运行结果；
3. 截图环境校验成功、版本输出正常的界面，全部插入实训报告。

---

## 模块二 ShopRec-CN官方四数据集获取与分文件读取

### 一、实训目的
1. 从零完成官方数据集下载、归档、整理，熟悉四份核心数据表结构；
2. 掌握大文件分块读取代码，解决百万级数据内存溢出问题；
3. 熟练使用数据探查代码，查看数据维度、字段、缺失值、数据样例。

### 二、数据集基本信息
- **数据集来源**：ModelScope平台 redmapler/ShopRec-CN 中文商城推荐专项数据集
- **数据集地址**：https://www.modelscope.cn/datasets/redmapler/ShopRec-CN
- **核心四文件及业务说明**：
  1. `users.csv`：用户基础数据表，主键id，存储用户画像、注册信息，用于用户分层、画像分析；
  2. `products.csv`：商品基础数据表，主键id，存储商品属性、价格、销量、标签等，是商品热度分析核心数据源；
  3. `browses.csv`：用户浏览行为表，通过user_id、product_id关联用户与商品，记录用户浏览行为数据；
  4. `orders.csv`：订单交易表，记录用户下单、消费、交易明细，用于营收分析、RFM建模。

### 三、分步实操步骤

#### 步骤1：下载并归档数据集
1. 打开数据集官方地址，点击「下载数据集」，获取四份CSV文件；
2. 检查文件名：`users.csv`、`products.csv`、`browses.csv`、`orders.csv`，无需改名；
3. 将四份文件全部放入项目 **data/raw** 文件夹中。

#### 步骤2：编写通用分块读取函数（适配超大文件）
新建Jupyter单元格，运行以下通用读取代码，解决内存溢出、编码报错问题：

```python
import pandas as pd

# 定义通用大文件读取函数
def read_big_csv(file_path, chunk_size=10000):
    chunk_list = []
    # 分块读取+容错编码
    for chunk in pd.read_csv(file_path, chunksize=chunk_size,
                             encoding="utf-8-sig", encoding_errors="ignore"):
        chunk_list.append(chunk)
    df = pd.concat(chunk_list, ignore_index=True)
    return df

# 读取四份原始数据
df_users = read_big_csv("./data/raw/users.csv")
df_products = read_big_csv("./data/raw/products.csv")
df_browses = read_big_csv("./data/raw/browses.csv")
df_orders = read_big_csv("./data/raw/orders.csv")

print("✅ 四份数据集全部读取完成！")
```

#### 步骤3：批量数据探查（零基础直接运行）
运行以下代码，一键输出所有数据表的核心信息：

```python
# 批量探查数据基本信息
def explore_data(df, name):
    print(f"==========【{name}数据探查结果】==========")
    print(f"数据行列数：{df.shape}")
    print("字段列表：")
    print(df.columns.tolist())
    print("前5行数据：")
    print(df.head())
    print("缺失值统计：")
    print(df.isnull().sum())
    print("\n")

# 依次探查四张表
explore_data(df_users, "用户表users")
explore_data(df_products, "商品表products")
explore_data(df_browses, "浏览表browses")
explore_data(df_orders, "订单表orders")
```

### 四、实训成果与截图要求
1. 截图data/raw目录下四份完整数据集文件；
2. 截图全部数据探查结果（行列数、字段、缺失值、样例数据）；
3. 手写四表关联关系：用户id关联订单、浏览表user_id，商品id关联订单、浏览表product_id。

---

# 第二阶段 数据预处理

## 模块三 大数据清洗Ⅰ——四文件分表降噪与异常处理

### 一、实训目的
1. 掌握分表独立清洗逻辑，针对不同业务数据表制定专属清洗规则；
2. 熟练完成去重、缺失值删除、异常值过滤三大核心清洗操作；
3. 生成清洗日志，清晰记录数据清洗前后变化，实现过程可追溯。

### 二、分步实操步骤

#### 步骤1：定义数据清洗日志工具
```python
def clean_log(origin_df, clean_df, name):
    print(f"===== {name}清洗日志 =====")
    print(f"清洗前数据量：{origin_df.shape[0]} 行")
    print(f"清洗后数据量：{clean_df.shape[0]} 行")
    print(f"去除无效数据：{origin_df.shape[0] - clean_df.shape[0]} 行\n")

# 备份原始数据
df_users_origin = df_users.copy()
df_products_origin = df_products.copy()
df_browses_origin = df_browses.copy()
df_orders_origin = df_orders.copy()
```

#### 步骤2：用户表清洗
```python
# 1.主键去重
df_users = df_users.drop_duplicates(subset=["id"], keep="first")
# 2.过滤异常年龄（0-100岁为合理区间）
if "age" in df_users.columns:
    df_users = df_users[(df_users["age"] >= 0) & (df_users["age"] <= 100)]
# 输出清洗日志
clean_log(df_users_origin, df_users, "用户表")
```

#### 步骤3：商品表清洗
```python
# 1.主键去重
df_products = df_products.drop_duplicates(subset=["id"], keep="first")
# 2.过滤负价格、负销量异常数据
if "price" in df_products.columns and "sales" in df_products.columns:
    df_products = df_products[(df_products["price"] > 0) & (df_products["sales"] >= 0)]
clean_log(df_products_origin, df_products, "商品表")
```

#### 步骤4：浏览表清洗
```python
# 1.核心字段去空
key_cols = ["user_id", "product_id", "create_time"]
df_browses = df_browses.dropna(subset=key_cols)
# 2.去除重复浏览记录
df_browses = df_browses.drop_duplicates(keep="first")
clean_log(df_browses_origin, df_browses, "浏览表")
```

#### 步骤5：订单表清洗
```python
# 1.核心字段去空
order_key = ["user_id", "product_id", "total_amount", "quantity"]
df_orders = df_orders.dropna(subset=order_key)
# 2.过滤负金额、负数量异常订单
df_orders = df_orders[(df_orders["total_amount"] > 0) & (df_orders["quantity"] > 0)]
# 3.订单去重
df_orders = df_orders.drop_duplicates(keep="first")
clean_log(df_orders_origin, df_orders, "订单表")
```

#### 步骤6：保存清洗后数据至clean目录
```python
# 批量保存清洗数据
df_users.to_csv("./data/clean/clean_users.csv", index=False, encoding="utf-8-sig")
df_products.to_csv("./data/clean/clean_products.csv", index=False, encoding="utf-8-sig")
df_browses.to_csv("./data/clean/clean_browses.csv", index=False, encoding="utf-8-sig")
df_orders.to_csv("./data/clean/clean_orders.csv", index=False, encoding="utf-8-sig")

print("✅ 所有数据表清洗完成，已保存至data/clean目录！")
```

### 三、实训成果要求
1. 截图四份数据表的清洗日志，清晰展示数据删减数量；
2. 打开data/clean目录，截图四份清洗后的文件；
3. 在报告中简单说明每类异常数据的剔除意义。

---

## 模块四 大数据清洗Ⅱ——时间特征衍生与四表数据融合

### 一、实训目的
1. 掌握时间字段标准化转换，自动衍生小时、月份等可分析时间特征；
2. 掌握多表左关联核心逻辑，从零搭建电商全维度特征宽表；
3. 生成建模专用标准数据集，为后续分析、可视化、建模提供数据源。

### 二、分步实操步骤

#### 步骤1：读取清洗后的标准数据
```python
df_users = pd.read_csv("./data/clean/clean_users.csv", encoding="utf-8-sig")
df_products = pd.read_csv("./data/clean/clean_products.csv", encoding="utf-8-sig")
df_browses = pd.read_csv("./data/clean/clean_browses.csv", encoding="utf-8-sig")
df_orders = pd.read_csv("./data/clean/clean_orders.csv", encoding="utf-8-sig")
print("✅ 清洗数据读取完成！")
```

#### 步骤2：时间字段标准化与特征衍生
```python
# 订单表时间处理
df_orders["created_at"] = pd.to_datetime(df_orders["created_at"])
df_orders["order_hour"] = df_orders["created_at"].dt.hour   # 小时特征
df_orders["order_month"] = df_orders["created_at"].dt.month # 月份特征

# 浏览表时间处理
df_browses["create_time"] = pd.to_datetime(df_browses["create_time"])
df_browses["browse_hour"] = df_browses["create_time"].dt.hour

print("✅ 时间特征衍生完成！")
```

#### 步骤3：多表逐层融合（左关联，保留所有订单数据）
```python
# 1.订单表关联商品表
df_merge1 = pd.merge(df_orders, df_products, left_on="product_id", right_on="id", how="left")
# 2.关联用户表
df_merge2 = pd.merge(df_merge1, df_users, left_on="user_id", right_on="id", how="left")
# 3.关联浏览行为表
df_full = pd.merge(df_merge2, df_browses, on=["user_id", "product_id"], how="left")

print(f"融合后宽表数据维度：{df_full.shape}")
print("宽表字段预览：")
print(df_full.columns.tolist())
```

#### 步骤4：保存最终特征宽表
```python
df_full.to_csv("./data/feature/full_feature_data.csv", index=False, encoding="utf-8-sig")
print("✅ 全维度特征宽表已保存至data/feature目录！")
df_full.head()
```

### 三、实训成果要求
1. 截图时间特征衍生成功的运行结果；
2. 截图宽表维度数据、字段列表、前5行数据；
3. 理解左关联作用：保留所有订单数据，匹配对应的商品、用户、浏览信息。

---

# 第三阶段 数据可视化分析

## 模块五 大数据EDA探索性商业分析

### 一、实训目的
1. 完成数据统计、分布分析、交叉分析全套EDA流程；
2. 从数据中挖掘用户行为规律、商品热度、消费特征；
3. 输出可直接用于电商运营的业务结论。

### 二、分步实操步骤

#### 步骤1：读取特征宽表
```python
df = pd.read_csv("./data/feature/full_feature_data.csv", encoding="utf-8-sig")
print("特征宽表读取成功！")
```

#### 步骤2：基础数值统计分析
```python
# 核心数值字段统计
print("===== 消费与销量核心统计 =====")
print(df[["total_amount", "quantity", "price"]].describe())
```

#### 步骤3：商品类目热度统计
```python
print("===== 各商品类目订单热度分布 =====")
cate_count = df["category_l1"].value_counts()
print(cate_count)
```

#### 步骤4：用户下单时段规律分析
```python
print("===== 各时段下单量统计 =====")
hour_analyse = df.groupby("order_hour")["user_id"].count().reset_index()
hour_analyse.columns = ["下单小时", "下单数量"]
print(hour_analyse)
```

#### 步骤5：类目营收交叉分析
```python
# 各类目订单量、总销售额、平均客单价
cate_pivot = pd.pivot_table(
    df, index="category_l1",
    values=["total_amount", "quantity"],
    aggfunc=["count", "sum", "mean"]
)
print("===== 类目营收交叉分析 =====")
print(cate_pivot)
```

### 三、实训成果要求
1. 截图所有EDA统计结果；
2. 自行总结3条以上业务结论（例：某类目销量最高、晚间为下单高峰期等）；
3. 将结论与数据一一对应，写入实训报告。

---

## 模块六 大数据静态可视化分析

### 一、实训目的
1. 熟练使用Matplotlib、Seaborn绘制标准可视化图表；
2. 实现类目热度、时段流量、热销商品可视化呈现；
3. 保存高清图表并完成图表业务解读。

### 二、分步实操步骤

#### 步骤1：全局绘图配置
```python
import matplotlib.pyplot as plt
import seaborn as sns
# 全局中文+负号修复
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(12, 6))
```

#### 步骤2：绘制商品类目热度柱状图
```python
cate_data = df["category_l1"].value_counts()
sns.barplot(x=cate_data.index, y=cate_data.values)
plt.title("各商品类目订单热度分布图", fontsize=14)
plt.xlabel("商品类目")
plt.ylabel("订单数量")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("./img/类目热度分布图.png", dpi=300)
plt.show()
```

#### 步骤3：绘制下单时段趋势图
```python
hour_data = df.groupby("order_hour")["user_id"].count()
hour_data.plot(kind="line", marker="o", color="#ff6b6b", linewidth=2)
plt.title("全天用户下单时段趋势图", fontsize=14)
plt.xlabel("小时")
plt.ylabel("下单量")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./img/下单时段趋势图.png", dpi=300)
plt.show()
```

### 三、实训成果要求
1. 两张高清图表自动保存至img目录；
2. 截图图表运行效果，附带文字解读插入报告。

---

## 模块七 交互式可视化与文本词云可视化数据分析

### 一、实训目的
1. 掌握Pyecharts交互式图表绘制，实现鼠标悬浮查看数据；
2. 掌握商品标签文本清洗、分词、词云生成全流程；
3. 多维度可视化挖掘商品卖点与平台特征。

### 二、分步实操步骤

#### 步骤1：交互式类目柱状图
```python
from pyecharts.charts import Bar
from pyecharts import options as opts

cate_data = df["category_l1"].value_counts()
bar = (
    Bar()
    .add_xaxis(list(cate_data.index))
    .add_yaxis("类目订单量", list(cate_data.values))
    .set_global_opts(title_opts=opts.TitleOpts(title="商品类目销量交互式分析图"))
)
bar.render("./img/交互式类目销量图.html")
print("✅ 交互式图表已生成至img目录！")
```

#### 步骤2：商品标签词云图生成
```python
from wordcloud import WordCloud

# 拼接所有商品标签
text = ""
tags_series = df_products["tags"].dropna()
for tag in tags_series:
    text += " ".join(str(tag).split(",")) + " "

# 生成词云
wc = WordCloud(
    width=1000, height=600,
    background_color="white",
    font_path="C:/Windows/Fonts/simhei.ttf"
).generate(text)

plt.imshow(wc)
plt.axis("off")
plt.savefig("./img/商品标签词云.png", dpi=300, bbox_inches="tight")
plt.show()
```

### 三、实训成果要求
1. 打开HTML交互式文件，截图悬浮交互效果；
2. 截图词云图，分析平台主流商品卖点。

---

# 第四阶段 数据建模与预测

## 模块八 基于RFM的用户价值分层建模

### 一、实训目的
1. 理解R、F、M三大指标业务含义；
2. 从零计算用户RFM指标、标准化数据、K-Means聚类；
3. 实现用户自动分层并匹配运营策略。

### 二、分步实操步骤

#### 步骤1：计算RFM三大指标
```python
# 时间转换
df_orders["created_at"] = pd.to_datetime(df_orders["created_at"])
max_date = df_orders["created_at"].max()

# 计算RFM
rfm = df_orders.groupby("user_id").agg(
    R=("created_at", lambda x: (max_date - x.max()).days),
    F=("id", "count"),
    M=("total_amount", "sum")
).reset_index()
print("RFM指标计算完成，预览：")
print(rfm.head())
```

#### 步骤2：数据标准化+K-Means聚类
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 标准化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["R", "F", "M"]])

# Kmeans四分类
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

# 统计各类用户均值
print("===== 用户分层结果 =====")
print(rfm.groupby("cluster")[["R", "F", "M"]].mean())
```

### 三、实训成果要求
1. 截图RFM指标计算结果、用户分层均值结果；
2. 自行定义四类用户：高价值用户、潜力用户、沉睡用户、流失用户，并撰写运营策略。

---

## 模块九 商品热度挖掘与销量预测建模

### 一、实训目的
1. 掌握多权重商品热度计算方法；
2. 实现随机森林销量预测建模与模型评估；
3. 输出商品热度榜单与模型精度指标。

### 二、分步实操步骤

#### 步骤1：计算商品综合热度
```python
# 统计商品核心指标
item_data = df.groupby("product_id").agg(
    浏览量=("user_id", "count"),
    销量=("quantity", "sum"),
    销售额=("total_amount", "sum")
).reset_index()

# 行业标准权重计算综合热度
item_data["综合热度"] = item_data["浏览量"]*0.2 + item_data["销量"]*0.5 + item_data["销售额"]*0.3
# 热度排序
item_hot_rank = item_data.sort_values("综合热度", ascending=False)
print("===== 商品综合热度TOP10 =====")
print(item_hot_rank.head(10))
```

#### 步骤2：随机森林销量预测建模
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 构建特征与标签
X = df[["order_hour", "price"]]
y = df["quantity"]

# 模型训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 预测与评估
y_pred = model.predict(X)
print(f"模型拟合优度R²：{r2_score(y, y_pred):.2f}")
print(f"平均绝对误差MAE：{mean_absolute_error(y, y_pred):.2f}")
```

### 三、实训成果要求
1. 截图商品热度TOP10榜单；
2. 截图模型评估指标，分析模型拟合效果。

---

## 模块十 基于用户行为的协同过滤推荐模型

### 一、实训目的
1. 理解协同过滤推荐算法原理；
2. 构建用户-商品相似度矩阵；
3. 实现个性化商品推荐并验证效果。

### 二、分步实操步骤

#### 步骤1：构建用户-商品交互矩阵
```python
# 构建透视矩阵
user_item = pd.pivot_table(df, index="user_id", columns="product_id", values="quantity", fill_value=0)
print(f"用户商品矩阵维度：{user_item.shape}")
```

#### 步骤2：计算用户相似度
```python
from sklearn.metrics.pairwise import cosine_similarity

user_sim = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)
print("用户相似度矩阵构建完成！")
```

#### 步骤3：编写推荐函数并测试
```python
def recommend_goods(user_id, top_n=5):
    # 找最相似用户
    sim_users = user_sim_df[user_id].sort_values(ascending=False)[1:top_n+1].index
    # 统计相似用户热门商品
    rec_items = df[df["user_id"].isin(sim_users)]["product_id"].value_counts().head(top_n)
    return rec_items.index.tolist()

# 随机测试
test_user = df["user_id"].iloc[10]
rec_list = recommend_goods(test_user)
print(f"用户{test_user}个性化推荐商品ID：{rec_list}")
```

### 三、实训成果要求
1. 截图推荐结果；
2. 分析推荐结果合理性，撰写模型优化思路。

---

# 第五阶段 项目总结

## 模块十一 项目工程化封装与代码优化

### 一、实训目的
1. 掌握Python项目模块化拆分思路；
2. 封装通用工具函数，优化代码结构；
3. 实现项目标准化、可复用、可落地。

### 二、分步实操步骤

**步骤1：新建utils.py工具文件（存入code目录）**  
将数据读取、日志输出、绘图配置等通用代码封装为工具函数；

**步骤2：代码模块化拆分**
1. `data_process.py`：存放所有数据读取、清洗、融合代码；
2. `visualization.py`：存放所有可视化绘图代码；
3. `model_train.py`：存放RFM、销量预测、推荐模型代码；

**步骤3：编写main.py主程序**  
依次调用各模块函数，实现一键运行全流程项目；

**步骤4：补充异常处理**  
为文件读取、模型训练代码添加try-except异常捕获，提升代码稳定性。

### 三、实训成果要求
1. 截图模块化文件目录；
2. 运行main.py实现一键执行全流程，截图运行成功界面。

---

## 模块十二 实训报告撰写与项目答辩

### 一、实训交付清单
1. **代码成果**：模块化全套源码；
2. **数据成果**：清洗数据、特征宽表、模型结果数据；
3. **可视化成果**：所有静态图、交互式图表、词云图；
4. **文档成果**：完整实训报告、答辩PPT、全部过程截图、日志文件。

### 二、报告标准结构
项目概述、环境搭建、数据集介绍、数据预处理、EDA分析、可视化成果、用户分层建模、销量预测、个性化推荐、工程化优化、实训总结、附录代码。

### 三、考核标准
过程成果30% + 代码质量35% + 分析建模成果20% + 报告答辩15%

---

## 附录 常见报错与一键解决方案

1. **中文乱码**：已全局配置SimHei字体，无需重复设置；
2. **内存溢出**：使用代码中分块读取，低配电脑可将chunk_size改为5000；
3. **词云字体报错**：检查字体路径，替换为系统自带simhei.ttf；
4. **库导入失败**：重启Jupyter内核，重新激活虚拟环境；
5. **文件路径报错**：全程使用相对路径，文件夹无中文无空格。
