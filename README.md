# GNNLab 数据预处理

本仓库提供统一的数据预处理脚本，用于将 `dsg/` 和 `ssg/` 目录中的原始骨架数据转换为 NumPy 格式，并将处理结果保存至 `processed/` 目录。**`main` 分支仅介绍项目的目录结构、数据准备方式和通用预处理流程**，不涉及具体的实验要求；各项实验的任务定义、数据处理细节和执行要求，请参阅对应实验分支中的说明。

## 项目结构

在完成原始数据解压并运行预处理后，项目的主要目录结构应如下所示：

```text
gnnlab/
├── dsg/                         # DSG 动态骨架原始数据
├── ssg/                         # SSG 静态骨架原始数据
├── processed/                   # 预处理生成的 NumPy 数据、标签和元数据
│   ├── dsg/                     # DSG 预处理结果
│   └── ssg/                     # SSG 预处理结果
├── tools/                       # 数据预处理工具
│   ├── preprocess_datasets.py   # DSG 和 SSG 的统一预处理入口
│   ├── export_dsg_source.py     # DSG 原始骨架导出工具
│   ├── organize_dsg_source.py   # DSG 数据划分整理工具
│   └── preprocessing/           # DSG、SSG 及通用预处理实现
├── .env.example                 # 本地路径配置模板
└── README.md                    # main 分支使用说明
```

## 分支说明

`main` 是公共的数据预处理入口，只说明原始数据准备、路径配置和预处理命令。实验内容按任务拆分到以下分支：

1. [`ssg`](../../tree/ssg)：**静态骨架图实验**，说明如何用关节点和连接表示人体姿态，以及坐姿与站姿二分类的数据范围和处理要求。
2. [`dsg`](../../tree/dsg)：**动态骨架图实验**，说明为什么动作识别需要时间信息，以及五类动态动作的数据范围、划分协议和处理要求。
3. [`ntu60-graph-models`](../../tree/ntu60-graph-models)：**自由研究问题**，自由文体不做具体约束，建议探索 Graph Transformer和GCN 能否在完整的 [NTU RGB+D 60](https://rose1.ntu.edu.sg/dataset/actionRecognition/) 骨架动作识别任务上提高准确率。

切换到对应分支即可查看该实验的独立 README，例如切换到`ssg`分支可以使用如下命令：

```bash
git switch ssg
```

## 环境要求

- Python 3.10 或更高版本
- NumPy

## 准备原始数据

原始数据统一打包在 `datasets.tar.gz` 中：

- 百度网盘：[下载 datasets.tar.gz](https://pan.baidu.com/s/1sc_oOfOLmiM4-UmVUGhyPA?pwd=q4am)
- 提取码：`q4am`

将压缩包放在项目根目录并解压：

```bash
tar -xzf datasets.tar.gz
```

解压后，项目根目录应包含 `dsg/` 和 `ssg/`。

默认情况下，脚本会以项目根目录为基础查找原始数据并保存处理结果。如需将数据或输出放在其他位置，可复制 `.env.example` 并修改其中的路径：

```bash
cp .env.example .env
```

`.env` 可按以下示例进行配置，通常只需将 `GNNLAB_ROOT` 修改为本项目的实际路径。各变量含义如下：

```dotenv
GNNLAB_ROOT=/path/to/gnnlab
GNNLAB_DSG_DIR=${GNNLAB_ROOT}/dsg
GNNLAB_SSG_DIR=${GNNLAB_ROOT}/ssg
GNNLAB_PROCESSED_DIR=${GNNLAB_ROOT}/processed
```

- `GNNLAB_ROOT`：GNNLab 项目根目录
- `GNNLAB_DSG_DIR`：动态骨架图实验数据集目录
- `GNNLAB_SSG_DIR`：静态骨架图实验数据集目录
- `GNNLAB_PROCESSED_DIR`：预处理结果输出目录

## 运行预处理

### 快速开始

在项目根目录运行以下命令，默认预处理 DSG 和 SSG 两套数据集：

```bash
python3 tools/preprocess_datasets.py
```

### 预处理指定数据集

如需仅预处理其中一套数据集，可运行：

```bash
python3 tools/preprocess_datasets.py --datasets ssg
python3 tools/preprocess_datasets.py --datasets dsg
```

预处理结果分别保存在 `processed/ssg/` 和 `processed/dsg/`。如果结果文件已存在，可添加 `--overwrite` 重新生成并覆盖已有结果：

```bash
python3 tools/preprocess_datasets.py --datasets all --overwrite
```

### 清理预处理结果

可按数据集清理已有的预处理结果：

```bash
python3 tools/preprocess_datasets.py --clean --datasets ssg
python3 tools/preprocess_datasets.py --clean --datasets dsg
python3 tools/preprocess_datasets.py --clean --datasets all
```

默认情况下，清理操作仅删除 `processed/` 下所选数据集的结果，不会影响 `dsg/` 和 `ssg/` 中的原始数据。更多参数请运行：

```bash
python3 tools/preprocess_datasets.py --help
```
