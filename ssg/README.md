# 静态姿势 JSON 数据集

本目录的数据来自 POLAR（Posture-level Action Recognition）数据集，从原始 9 个姿态类别中筛选出 `sitting` 和 `standing` 两类，并已完成训练集/测试集划分。目录中保存的是 OpenPose JSON；每个 JSON 文件都是一个独立的单帧样本。

## 目录结构

```text
datasets/
├── README.md
├── sitting/
│   ├── train/       # 坐姿训练集，标签 0
│   │   └── *.json
│   └── test/        # 坐姿测试集，标签 0
│       └── *.json
└── standing/
    ├── train/       # 站姿训练集，标签 1
    │   └── *.json
    └── test/        # 站姿测试集，标签 1
        └── *.json
```

数据按照 8:2 划分：每一类 80% 用于训练，20% 用于测试。当前每一类有 240 个训练 JSON 和 60 个测试 JSON。

类别由文件所在的目录决定：

```text
datasets/sitting/train/*.json  -> sitting -> 0
datasets/sitting/test/*.json   -> sitting -> 0
datasets/standing/train/*.json -> standing -> 1
datasets/standing/test/*.json  -> standing -> 1
```

请不要把测试集文件放进训练目录，也不要在训练完成后根据测试集结果反复修改模型参数。测试集用于最后评价模型的泛化能力。

## 文件名

文件名示例：

```text
all__p1_05942__sitting__000_keypoints.json
```

各部分可以理解为：

```text
all       __ p1_05942 __ sitting __ 000 __ keypoints.json
来源/集合     样本编号      姿势类别   帧编号    OpenPose关键点文件
```

- `all`：原始数据来源或集合名称。
- `p1_05942`：原始样本或主体编号，用于追踪样本来源。
- `sitting`：文件名中记录的类别；实际训练标签以所在目录为准。
- `000`：帧编号。本数据是单帧数据，通常为 `000`，不表示一个视频序列。
- `keypoints.json`：OpenPose 输出文件。

## JSON 内部结构

每个文件的顶层结构如下：

```json
{
  "version": 1.3,
  "people": [
    {
      "person_id": [-1],
      "pose_keypoints_2d": [
        0, 0, 0,
        21.8319, 44.5065, 0.690678,
        35.4986, 46.8212, 0.575722
      ],
      "face_keypoints_2d": [],
      "hand_left_keypoints_2d": [],
      "hand_right_keypoints_2d": [],
      "pose_keypoints_3d": [],
      "face_keypoints_3d": [],
      "hand_left_keypoints_3d": [],
      "hand_right_keypoints_3d": []
    }
  ]
}
```

上面只展示了 `pose_keypoints_2d` 的前 3 个关节，实际数组包含 25 个关节，共 75 个数。

### 顶层字段

- `version`：OpenPose JSON 格式版本，当前数据通常为 `1.3`。
- `people`：检测到的人体列表。当前数据通常只有一个人。

### `people` 中的人体字段

- `person_id`：人体 ID，不作为分类输入。
- `pose_keypoints_2d`：BODY_25 人体二维骨架关键点，是本项目使用的主要字段。
- `face_keypoints_2d`：脸部二维关键点，本项目不使用。
- `hand_left_keypoints_2d`：左手二维关键点，本项目不使用。
- `hand_right_keypoints_2d`：右手二维关键点，本项目不使用。
- 各个以 `_3d` 结尾的字段：三维关键点，本数据中通常为空，本项目不使用。

## `pose_keypoints_2d` 数组

数组按照下面的方式重复保存 25 个关节：

```text
[x, y, confidence, x, y, confidence, ...]
```

对第 `i` 个关节（编号从 0 开始）：

```text
x          = pose_keypoints_2d[3*i]
y          = pose_keypoints_2d[3*i + 1]
confidence = pose_keypoints_2d[3*i + 2]
```

- `x`：图像横坐标，通常以像素为单位。
- `y`：图像纵坐标，通常以像素为单位，向下为正方向。
- `confidence`：OpenPose 的检测置信度，接近 0 通常表示该点没有被可靠检测到。

## BODY_25 关节顺序

| 编号 | 关节 | 编号 | 关节 |
|---:|---|---:|---|
| 0 | Nose 鼻子 | 13 | L Knee 左膝 |
| 1 | Neck 颈部 | 14 | L Ankle 左脚踝 |
| 2 | R Shoulder 右肩 | 15 | R Eye 右眼 |
| 3 | R Elbow 右肘 | 16 | L Eye 左眼 |
| 4 | R Wrist 右手腕 | 17 | R Ear 右耳 |
| 5 | L Shoulder 左肩 | 18 | L Ear 左耳 |
| 6 | L Elbow 左肘 | 19 | L Big Toe 左大脚趾 |
| 7 | L Wrist 左手腕 | 20 | L Small Toe 左小脚趾 |
| 8 | MidHip 髋部中心 | 21 | L Heel 左脚跟 |
| 9 | R Hip 右髋 | 22 | R Big Toe 右大脚趾 |
| 10 | R Knee 右膝 | 23 | R Small Toe 右小脚趾 |
| 11 | R Ankle 右脚踝 | 24 | R Heel 右脚跟 |
| 12 | L Hip 左髋 | | |

## 后续处理

同学需要自行完成以下步骤：

1. 读取 `sitting/train` 和 `standing/train` 作为训练数据。
2. 读取 `sitting/test` 和 `standing/test` 作为测试数据。
3. 对关键点进行对齐、中心化和尺度归一化。
4. 将 JSON 转换成模型需要的训练格式。
5. 使用训练集训练模型，最后只在测试集上进行评价。

本目录只提供原始 JSON，不预先提供 `.npy`、`.pkl` 或模型文件。

## 预处理命令

在 GNNLab 项目根目录执行：

```bash
python3 tools/preprocess_datasets.py --datasets ssg --overwrite
```

结果写入 `processed/ssg/`，默认只生成训练和测试的
`data.npy`、`label.npy` 以及一个 `metadata.json`。BODY_25 的关节名称和边保存在
metadata 中，不再额外生成多套图结构或调试文件。

坐标以有效髋部关节的中心为原点，将躯干方向旋转至竖直方向，并使用肩部和髋部
骨段长度的中位数归一化尺度。该方法保留四肢相对躯干的伸展程度。当前 3 个
`standing/train` JSON 没有检测到人体，会被跳过，因此标准输出为 477 个训练样本
和 120 个测试样本。
