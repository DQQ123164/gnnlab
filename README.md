# 静态骨架图实验

本实验是课程中的第一个基础任务：将单帧人体关节表示为静态骨架图，并完成坐姿与站姿的二分类。实验不要求一开始就设计复杂或高度完善的姿态识别方案，重点是理解图卷积神经网络（GCN）如何利用关节之间的连接关系聚合节点特征，并将整张骨架图编码为可用于分类的图级特征向量。整体任务规模较小，难度较低。

## 问题引入：人体姿态如何表示为图？

一张姿态图像可以用关节点和它们之间的连接来描述：关节点是图的节点，连接是边。下图在坐姿与站姿照片上叠加了骨架，展示同一人体结构在不同姿态下的形状差异。

| 坐姿 | 站姿 |
|:---:|:---:|
| <img src="images/all__p1_11777__sitting__000_rendered.png" alt="坐姿照片上的人体关节点与连线" height="360"> | <img src="images/all__p1_22876__standing__000_rendered.png" alt="站姿照片上的人体关节点与连线" height="360"> |

图片仅用于说明图表示；预处理输出的是关节坐标与置信度，而非原始照片。关节名称和连接关系记录在生成的 `processed/ssg/metadata.json` 中。

## 实验目的

- **实验要求**：使用预处理后的单帧`BODY25`骨架完成坐姿与站姿二分类。**主模型必须使用图神经网络**：以 25 个关节为节点，以 `metadata.json` 中的`BODY25`连接关系为边，以归一化后的二维坐标和置信度为节点特征；模型**至少包含一层图卷积或等价的邻域消息传递，通过图级池化（readout）得到整张骨架图的特征向量**，再由分类层输出 `sitting` 或 `standing`。本实验只使用单帧静态信息，不涉及时序建模。

- **建模限制**：原始 `.json`文件和生成的 `.npy` 文件仅作为数据来源与存储接口。训练主模型前，必须将每个样本组织为包含节点特征和边连接的图；**不得跳过图构建**，将关键点或数组直接展平后交给传统机器学习模型、MLP 或 CNN 作为主方案。非图模型可以作为对比基线，但**不能替代主图模型**。

- **评判标准**：最终以固定测试集上的 Top-1 Accuracy 作为主要指标。Top-1 Accuracy 达到 **80% 及以上**，即视为完整完成本实验要求；在此基础上，准确率越高表示分类效果越好。同时报告 `sitting`、`standing` 两类的 Precision、Recall、F1 和混淆矩阵；当模型准确率接近时，以 Macro-F1 辅助判断两类是否都得到稳定识别。模型之间只能在相同训练集、测试集和预处理条件下比较。

## 数据范围与标签

SSG 来源于 POLAR（Posture-level Action Recognition）数据集，只使用原始 9 个姿态类别中的 `sitting` 和 `standing`，并非完整的 POLAR 数据集。

| 标签 | 姿势类别 | train | test |
|---:|---|---:|---:|
| 0 | sitting | 240 | 60 |
| 1 | standing | 237 | 60 |
| **总计** | **2 类** | **477** | **120** |

原始 standing 训练集有 240 个 JSON，其中 3 个没有检测到人体，预处理时跳过，所以生成 237 个 standing 训练样本。

## 输入与预处理

- 输入目录：`ssg/sitting/{train,test}/*.json`、`ssg/standing/{train,test}/*.json`。
- 每个样本使用 OpenPose BODY_25 关节点；存在多个人时选择关节点置信度总和最高的人。
- 对有效关节点进行中心、朝向和尺度归一化；无效关节点的坐标及置信度设为 0。没有有效关节点的样本跳过。
- 保持源数据的 `train`、`test` 划分，不再额外随机切分。

## 快速开始

准备好原始数据后，在项目根目录执行：

```bash
python3 tools/preprocess_datasets.py --datasets ssg
```

已有输出需要重新生成时，添加 `--overwrite`；可以用 `--ssg-min-confidence` 设置参与归一化的最低关节点置信度。

## 输出接口

结果保存在 `processed/ssg/`：

```text
processed/ssg/
|-- metadata.json
|-- train_data.npy
|-- train_label.npy
|-- test_data.npy
`-- test_label.npy
```

- `train_data.npy` / `test_data.npy`：`float32` 骨架特征，形状为 `(N, 25, 3)`，最后一维依次为归一化后的 `x、y` 坐标和置信度。
- `train_label.npy` / `test_label.npy`：与数据逐样本对应的 `int64` 标签，`0` 表示 `sitting`，`1` 表示 `standing`。
- `metadata.json`：记录数据统计和 `BODY25` 骨架信息；模型应从 `body25.edges` 读取连接关系并构建图。
