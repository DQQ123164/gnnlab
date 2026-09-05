# GNNLab 实验数据接口

本实验统一管理两套骨架数据，并将它们预处理成可被下游训练项目直接读取的稳定数据接口：

- `dsg`：面向五分类动态骨架数据。
- `ssg`：面向二分类（坐姿/站姿）静态骨架数据。

原始数据保存在 `dsg/` 和 `ssg/`，统一预处理结果保存在 `processed/`。下游项目应读取 `processed/`，不直接依赖源数据目录的内部结构。

## 数据集类别与样本数

### DSG 五分类动态骨架数据

DSG 来源于 NTU RGB+D 60 骨架数据集，从原始 60 个动作类别中筛选出 `A033`、`A044`、`A045`、`A046` 和 `A047` 共 5 类，并分别提供 `xsub` 和 `xview` 两种数据划分。DSG 不是完整的 NTU RGB+D 60 数据集：

| 标签 | 动作类别 | xsub train | xsub test | xview train | xview test |
|---:|---|---:|---:|---:|---:|
| 0 | A033 check time | 536 | 134 | 504 | 126 |
| 1 | A044 headache | 536 | 134 | 504 | 126 |
| 2 | A045 chest pain | 537 | 134 | 505 | 126 |
| 3 | A046 back pain | 538 | 134 | 506 | 126 |
| 4 | A047 neck pain | 538 | 134 | 506 | 126 |
| **总计** | **5 类** | **2685** | **670** | **2525** | **630** |

### SSG 二分类静态骨架数据

SSG 来源于 POLAR（Posture-level Action Recognition）数据集，从原始 9 个姿态类别中筛选出 `sitting` 和 `standing` 两类。SSG 不是完整的 POLAR 数据集：

| 标签 | 姿势类别 | train | test |
|---:|---|---:|---:|
| 0 | sitting | 240 | 60 |
| 1 | standing | 237 | 60 |
| **总计** | **2 类** | **477** | **120** |

SSG 原始训练集包含 240 个 standing JSON，其中 3 个文件没有检测到人体，预处理时会跳过，因此生成的 standing 训练样本为 237 个。

## 数据下载

原始 DSG 和 SSG 数据统一打包在 `datasets.zip` 中：

- 文件名：`datasets.zip`
- 百度网盘：[点击下载 datasets.zip](https://pan.baidu.com/s/1Fb1u1TgfJWR3bqnDQjA31A?pwd=iy5s)
- 提取码：`iy5s`

下载并解压后，应将原始数据放在本项目的 `dsg/` 和 `ssg/` 目录中。

## 环境要求

- Python 3.10 或更高版本
- NumPy

## 项目结构

```text
gnnlab/
|-- README.md                         # 项目总说明
|-- tools/
|   `-- preprocess_datasets.py        # 统一预处理入口
|-- dsg/                              # 原始skeleton（需在百度网盘下载）
|   |-- README.md
|   |-- xsub/                         # Cross-Subject 协议
|   |   |-- train/*.skeleton
|   |   `-- test/*.skeleton
|   `-- xview/                        # Cross-View 协议
|       |-- train/*.skeleton
|       `-- test/*.skeleton
|-- ssg/                              # 原始JSON（需在百度网盘下载）
|   |-- README.md
|   |-- sitting/{train,test}/*.json   # 坐姿，标签 0
|   `-- standing/{train,test}/*.json  # 站姿，标签 1
`-- processed/                        # 统一生成的训练接口
    |-- dsg/
    |   |-- metadata.json
    |   |-- xsub/...
    |   `-- xview/...
    `-- ssg/
        |-- metadata.json
        |-- train_data.npy
        |-- train_label.npy
        |-- test_data.npy
        `-- test_label.npy
```

## 快速开始

在项目根目录执行：

```bash
python3 tools/preprocess_datasets.py
```

当前 `processed/` 已经存在时，如需重新生成，应显式使用 `--overwrite`：

```bash
python3 tools/preprocess_datasets.py --overwrite
```

## 常用命令

只处理 DSG：

```bash
python3 tools/preprocess_datasets.py --datasets dsg --overwrite
```

只处理 SSG：

```bash
python3 tools/preprocess_datasets.py --datasets ssg --overwrite
```

同时处理两套数据：

```bash
python3 tools/preprocess_datasets.py --datasets all --overwrite
```

一键删除全部已生成数据：

```bash
python3 tools/preprocess_datasets.py --clean --datasets all
```

只删除某一套生成数据：

```bash
python3 tools/preprocess_datasets.py --clean --datasets ssg
python3 tools/preprocess_datasets.py --clean --datasets dsg
```