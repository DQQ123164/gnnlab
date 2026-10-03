# GNNLab 数据预处理

本仓库提供统一的数据准备与预处理工具，将动态骨架图（DSG）和静态骨架图（SSG）数据转换为 NumPy 格式，并将结果保存到 `processed/`。**`main` 分支只介绍项目结构、数据准备方式和通用预处理流程**；各实验的任务定义、模型要求和评判标准请查看对应分支。

## 项目结构

完成原始数据准备和预处理后，项目的主要目录结构如下：

```text
gnnlab/
├── dsg/                          # 从 NTU60 原始压缩包生成的 DSG 官方划分
│   ├── xsub/{train,test}/        # Cross-Subject 训练集与测试集
│   └── xview/{train,test}/       # Cross-View 训练集与测试集
├── ssg/                          # SSG 静态骨架原始数据
├── processed/                    # NumPy 数据、标签和元数据
│   ├── dsg/                      # DSG 预处理结果
│   └── ssg/                      # SSG 预处理结果
├── tools/                        # 数据处理工具
│   ├── split_dsg_from_ntu.py     # 从 NTU60 原始 ZIP 筛选五类并生成官方划分
│   ├── preprocess_datasets.py    # DSG 和 SSG 的统一预处理入口
│   └── preprocessing/            # DSG、SSG 及通用预处理实现
├── .env.example                  # 本地路径配置模板
└── README.md                     # main 分支使用说明
```

## 分支说明

`main` 是公共的数据准备与预处理入口。实验内容按任务拆分到以下分支：

1. [`ssg`](../../tree/ssg)：**静态骨架图实验**，使用关节点及其连接表示人体姿态，完成坐姿与站姿二分类。
2. [`dsg`](../../tree/dsg)：**动态骨架图实验**，结合骨架的空间结构与时间信息，完成五类动作识别。
3. [`ntu60-graph-models`](../../tree/ntu60-graph-models)：**自由研究问题**，不限制具体文体，建议探索 Graph Transformer 和 GCN 能否提高完整 [NTU RGB+D 60](https://rose1.ntu.edu.sg/dataset/actionRecognition/) 骨架动作识别任务的准确率。

切换到对应分支即可查看独立实验说明，例如：

```bash
git switch ssg
```

## 环境要求

- Python 3.10 或更高版本
- NumPy

## 路径配置

默认情况下，脚本从项目根目录下的 `dsg/` 和 `ssg/` 读取数据，并将结果写入 `processed/`。如需修改路径，可复制配置模板：

```bash
cp .env.example .env
```

通常只需将 `GNNLAB_ROOT` 修改为项目的实际路径：

```dotenv
GNNLAB_ROOT=/path/to/gnnlab
GNNLAB_DSG_DIR=${GNNLAB_ROOT}/dsg
GNNLAB_SSG_DIR=${GNNLAB_ROOT}/ssg
GNNLAB_PROCESSED_DIR=${GNNLAB_ROOT}/processed
```

- `GNNLAB_ROOT`：GNNLab 项目根目录
- `GNNLAB_DSG_DIR`：DSG 官方划分的输出与读取目录
- `GNNLAB_SSG_DIR`：SSG 原始数据目录
- `GNNLAB_PROCESSED_DIR`：预处理结果输出目录

## 准备原始数据

### SSG 数据

SSG 原始数据继续通过课程数据包 `datasets.tar.gz` 提供：

- 百度网盘：[下载 datasets.tar.gz](https://pan.baidu.com/s/1sc_oOfOLmiM4-UmVUGhyPA?pwd=q4am)
- 提取码：`q4am`

课程数据包中仍包含旧版 `dsg/` 目录，当前 DSG 流程不再使用该目录。为避免旧划分与后续生成的新划分冲突，请将压缩包放在项目根目录，并只提取其中的 `ssg/`：

```bash
tar -xzf datasets.tar.gz ssg/
```

### DSG 数据

DSG 不再使用预先筛选的数据包，而是由脚本直接从 NTU RGB+D 官方原始骨架压缩包中筛选五类动作，并生成官方 `xsub` 和 `xview` 划分。

本实验只需要 `nturgbd_skeletons_s001_to_s017.zip`。该压缩包对应 NTU RGB+D 60 的 A001-A060；`nturgbd_skeletons_s018_to_s032.zip` 包含 NTU RGB+D 120 新增的 A061-A120，本实验不使用。

假设原始压缩包位于 `/home/dqq/NTU-DA/dataset/`，先执行只读校验，确认压缩包文件清单、五类样本数量和官方划分均符合预期：

```bash
python3 tools/split_dsg_from_ntu.py \
  --source-dir /home/dqq/NTU-DA/dataset \
  --dry-run
```

校验通过时应得到 4713 个有效样本（另排除 27 个已知无效样本），其中 `xsub` 为 3341 个训练样本和 1372 个测试样本，`xview` 为 3134 个训练样本和 1579 个测试样本。

校验通过后，正式生成 DSG 数据划分：

```bash
python3 tools/split_dsg_from_ntu.py \
  --source-dir /home/dqq/NTU-DA/dataset
```

如果原始数据位于其他位置，请将上述路径替换为实际目录，例如 `/path/to/NTU-DA/dataset`。脚本默认写入 `GNNLAB_DSG_DIR`，并生成以下目录：

```text
dsg/
├── xsub/
│   ├── train/
│   └── test/
└── xview/
    ├── train/
    └── test/
```

同一样本出现在两种协议中时，脚本默认使用硬链接以节省空间；如需生成相互独立的文件副本，可添加 `--copy-files`。如果输出目录中已有划分，需添加 `--overwrite` 才会重新生成。

划分完成后，将 DSG 骨架文件转换为模型可读取的 NumPy 数据：

```bash
python3 tools/preprocess_datasets.py --datasets dsg
```

## 运行预处理

完成上述数据准备后，在项目根目录运行统一预处理脚本。默认同时处理 DSG 和 SSG：

```bash
python3 tools/preprocess_datasets.py
```

也可以只处理指定数据集：

```bash
python3 tools/preprocess_datasets.py --datasets ssg
python3 tools/preprocess_datasets.py --datasets dsg
python3 tools/preprocess_datasets.py --datasets all
```

预处理结果分别保存在 `processed/ssg/` 和 `processed/dsg/`。如果结果文件已存在，可添加 `--overwrite` 重新生成：

```bash
python3 tools/preprocess_datasets.py --datasets all --overwrite
```

### 清理预处理结果

可按数据集清理 `processed/` 下已有的预处理结果：

```bash
python3 tools/preprocess_datasets.py --clean --datasets ssg
python3 tools/preprocess_datasets.py --clean --datasets dsg
python3 tools/preprocess_datasets.py --clean --datasets all
```

清理操作不会影响 `dsg/` 和 `ssg/` 中的源数据。更多参数请运行：

```bash
python3 tools/preprocess_datasets.py --help
python3 tools/split_dsg_from_ntu.py --help
```

## 许可证

本仓库自行编写的代码与文档采用 [MIT License](LICENSE)。

NTU RGB+D 数据集及其衍生数据不属于 MIT License 的授权范围。使用者须自行获取数据并遵守 [NTU RGB+D 官方使用条款](https://rose1.ntu.edu.sg/dataset/actionRecognition/)；本仓库不提供或再分发 NTU RGB+D 原始数据及其预处理结果。
