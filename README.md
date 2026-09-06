# GNNLab 实验数据接口

本实验统一管理两套骨架数据，并将它们预处理成稳定、通用的数据接口：

- `dsg`：五分类动态骨架序列数据。
- `ssg`：二分类（坐姿/站姿）静态骨架姿态数据。


## 数据集类别与样本数

### DSG 五分类动态骨架数据

DSG 来源于 NTU RGB+D 60 骨架数据集，从原始 60 个动作类别中筛选出 `A059`、`A030`、`A016`、`A005` 和 `A027` 共 5 类较容易区分的数据类别。预处理脚本从同一份完整原始数据按 NTU RGB+D 60 官方规则生成 Cross-Subject（`xsub`）和 Cross-View（`xview`）两种协议（DSG 不是完整的 NTU RGB+D 60 数据集）：

| 标签 | 动作类别 | xsub train | xsub test | xview train | xview test |
|----|---|---:|---:|---:|---:|
| 0 | A059 walking towards each other | 666 | 273 | 623 | 316 |
| 1 | A030 typing on a keyboard | 669 | 275 | 628 | 316 |
| 2 | A016 wear a shoe | 667 | 273 | 625 | 315 |
| 3 | A005 drop | 667 | 275 | 626 | 316 |
| 4 | A027 jump up | 672 | 276 | 632 | 316 |
| **总计** | **5 类** | **3341** | **1372** | **3134** | **1579** |

协议解释：`xsub` 以官方 20 名训练受试者划分训练集，其余受试者作为测试集；`xview` 以摄像机 2、3 的样本作为训练集，摄像机 1 的样本作为测试集。这里不再对官方训练侧做随机 8:2 二次切分。

### SSG 二分类静态骨架数据

SSG 来源于 POLAR（Posture-level Action Recognition）数据集，从原始 9 个姿态类别中筛选出 `sitting` 和 `standing` 两类。SSG 不是完整的 POLAR 数据集：

| 标签 | 姿势类别 | train | test |
|---:|---|---:|---:|
| 0 | sitting | 240 | 60 |
| 1 | standing | 237 | 60 |
| **总计** | **2 类** | **477** | **120** |

SSG 原始训练集包含 240 个 standing JSON，其中 3 个文件没有检测到人体，预处理时会跳过，因此生成的 standing 训练样本为 237 个。

## 数据下载

原始 DSG 和 SSG 数据统一打包在 `datasets.tar.gz` 中：

- 文件名：`datasets.tar.gz`
- 百度网盘：[点击下载 datasets.tar.gz](https://pan.baidu.com/s/1sc_oOfOLmiM4-UmVUGhyPA?pwd=q4am)
- 提取码：`q4am`

下载后，将`datasets.tar.gz`放在在项目根目录下，然后执行：

```bash
tar -xzf datasets.tar.gz
```

解压后应得到本项目所需的 `dsg/` 和 `ssg/` 原始数据目录。

## 路径配置

首次使用时复制环境变量模板：

```bash
mv .env.example .env
```

然后编辑 `.env`，将项目根目录改为本机实际位置：

```dotenv
GNNLAB_ROOT=/path/to/gnnlab
```

其余数据路径默认由 `GNNLAB_ROOT` 派生，一般不需要修改：

| 环境变量 | 用途 |
|---|---|
| `GNNLAB_ROOT` | 本项目根目录 |
| `GNNLAB_DSG_DIR` | DSG 原始数据目录 |
| `GNNLAB_SSG_DIR` | SSG 原始数据目录 |
| `GNNLAB_PROCESSED_DIR` | 统一预处理输出目录 |
| `NTU60_ANNOTATION_FILE` | 可选，仅从 NTU60 注释重新导出 DSG 时使用 |

## 环境要求

- Python 3.10 或更高版本
- NumPy

## 项目结构

```text
gnnlab/
|-- README.md                         # 项目总说明
|-- .env.example                     # 本地路径配置模板
|-- tools/
|   |-- export_dsg_source.py          # 从 NTU60 注释恢复正确的 DSG 原始骨架
|   |-- organize_dsg_source.py        # 实体整理官方 XSub/XView 原始目录
|   |-- preprocess_datasets.py        # 稳定的命令行入口与任务调度
|   `-- preprocessing/
|       |-- __init__.py               # 预处理包公开接口
|       |-- common.py                 # 原子写入、清理、校验和进度显示
|       |-- config.py                 # .env 加载与路径配置
|       |-- dsg.py                    # DSG skeleton 解析与输出
|       `-- ssg.py                    # SSG JSON 解析、归一化与输出
|-- dsg/                              # 原始skeleton（需在百度网盘下载）
|   |-- xsub/{train,test}/*.skeleton
|   `-- xview/{train,test}/*.skeleton
|-- ssg/                              # 原始JSON（需在百度网盘下载）
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

处理 DSG：

```bash
python3 tools/preprocess_datasets.py --datasets dsg --overwrite
```

处理 SSG：

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

删除某一套生成数据：

```bash
python3 tools/preprocess_datasets.py --clean --datasets ssg
python3 tools/preprocess_datasets.py --clean --datasets dsg
```
