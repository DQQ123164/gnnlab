# GNNLab 数据预处理

本仓库的预处理脚本将原始骨架数据转换为 `processed/` 下的 NumPy 数据接口。这里仅说明如何准备数据和运行脚本；各数据集的实验要求分别放在独立分支。

## 分支说明

`main` 是公共的数据预处理入口，只说明原始数据准备、路径配置和预处理命令。实验内容按任务拆分到以下分支：

1. [`ssg`](../../tree/ssg)：静态骨架图实验，说明如何用关节点和连接表示人体姿态，以及坐姿与站姿二分类的数据范围和处理要求。
2. [`dsg`](../../tree/dsg)：动态骨架图实验，说明为什么动作识别需要时间信息，以及五类动态动作的数据范围、划分协议和处理要求。
3. [`experiment-tbd`](../../tree/experiment-tbd)：为后续实验预留的文档分支，具体用途将在实验确定后补充。

切换到对应分支即可查看该实验的独立 README，例如：

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

路径默认相对于项目根目录。如需指定其他位置，可复制 `.env.example` 并修改 `GNNLAB_ROOT` 或对应数据集路径：

```bash
cp .env.example .env
```

`.env` 中的可用变量包括 `GNNLAB_ROOT`、`GNNLAB_DSG_DIR`、`GNNLAB_SSG_DIR` 和 `GNNLAB_PROCESSED_DIR`；不配置时使用项目根目录下的同名目录。

## 运行预处理

在项目根目录运行：

```bash
python3 tools/preprocess_datasets.py
```

默认处理两套数据。也可以只处理指定数据集：

```bash
python3 tools/preprocess_datasets.py --datasets ssg
python3 tools/preprocess_datasets.py --datasets dsg
```

生成结果保存在 `processed/ssg/` 和 `processed/dsg/`。如果目标文件已存在，重新生成时需要显式添加 `--overwrite`，例如：

```bash
python3 tools/preprocess_datasets.py --datasets all --overwrite
```

仅清理指定数据集的已生成结果：

```bash
python3 tools/preprocess_datasets.py --clean --datasets ssg
python3 tools/preprocess_datasets.py --clean --datasets dsg
```

使用 `--clean --datasets all` 可清理两套已生成结果；原始数据不受影响。运行 `python3 tools/preprocess_datasets.py --help` 可查看其他路径与处理选项。
