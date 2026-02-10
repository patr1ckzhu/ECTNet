# OpenBCI Cyton 数据采集指南

## 1. 硬件准备

### 你需要的东西

- OpenBCI Cyton 板 + USB Dongle
- 8 个盘状电极（dry/wet 均可）+ 2 个耳夹电极
- EEG 帽 或 Ten20 导电膏 + 卷尺（自行定位）
- 电脑（Windows，已装好环境）

### 电极接线

| Cyton 引脚 | 电极位置 | 10-20 名称 | 说明 |
|-----------|---------|-----------|------|
| N1P | 左侧运动皮层 | C3 | 左右手 MI 核心通道 |
| N2P | 右侧运动皮层 | C4 | 左右手 MI 核心通道 |
| N3P | 中线运动皮层 | Cz | 中线补充 |
| N4P | 中线额中央 | FCz | 运动准备区 |
| N5P | 左侧中央顶叶 | CP1 | 体感反馈区 |
| N6P | 右侧中央顶叶 | CP2 | 体感反馈区 |
| N7P | 左侧额中央 | FC3 | 运动规划区 |
| N8P | 右侧额中央 | FC4 | 运动规划区 |
| **SRB2** | 左耳垂/左乳突 | — | 参考电极 |
| **BIAS** | 右耳垂/右乳突 | — | 地电极 |

### 电极位置图

对照 BCI IV-2a 的 22 通道图，我们用的是中间区域（感觉运动皮层）的 8 个：

```
        (1 Fz)
   [2 FC3] (3) [4 FCz] (5) [6 FC4]
(7)  [8 C3]  (9) [10 Cz] (11) [12 C4] (13)
   (14) [15 CP1] (16) [17 CP2] (18)
        (19) (20 Pz) (21)
              (22)

[ ] = 我们使用的 8 个通道
( ) = 不使用
```

### 通道顺序说明

此顺序与代码中 `CHANNELS_8CH_NAMES = ['C3', 'C4', 'Cz', 'FCz', 'CP1', 'CP2', 'FC3', 'FC4']` 一致，兼容 A2 数据集预训练模型，方便迁移学习。

## 2. 软件环境

### 安装 OpenBCI GUI

1. 从 [OpenBCI Downloads](https://openbci.com/downloads) 下载 OpenBCI GUI
2. 安装并打开，选择 **Cyton (Serial/Dongle)**
3. 设置：8 通道，250 Hz 采样率

### Conda 环境

采集脚本使用 `acquisition` 环境（Python 3.10 + PsychoPy）：

```powershell
# 首次安装（已完成的话跳过）
conda create -n acquisition python=3.10 -y
conda activate acquisition
pip install psychopy pylsl numpy
```

训练使用 `eeg-moabb` 环境（Python 3.12 + PyTorch）。

## 3. 采集流程

### 第一步：启动 OpenBCI GUI 的 LSL 输出

1. 打开 OpenBCI GUI，连接 Cyton
2. 检查 8 个通道信号是否正常（无断连、无大幅 50Hz 干扰）
3. 进入 **Networking** 标签页 → 选择 **LSL**
4. Stream name 设为 `obci_eeg1`
5. 点 **Start** 开始推流

### 第二步：启动录制器

```powershell
conda activate acquisition
cd C:\Users\Patrick\PycharmProjects\ECTNet
python acquisition/recorder.py
```

等待输出 `EEG stream: 8 channels @ 250 Hz` 和 `Recording...`。

### 第三步：启动实验范式

新开一个终端：

```powershell
conda activate acquisition
cd C:\Users\Patrick\PycharmProjects\ECTNet
python acquisition/paradigm.py
```

- 屏幕显示指导语，按 **空格** 开始
- 30 个 trial（15 左手 + 15 右手），约 4.5 分钟
- 看到箭头后**想象**对应手的握拳动作，不要真的动
- 按 **ESC** 可中途退出

### 第四步：保存

实验结束后，回到 recorder 终端按 **Ctrl+C**，文件自动保存到：

```
acquisition/recordings/recording_YYYYMMDD_HHMMSS.npz
```

## 4. 数据转换

```powershell
conda activate acquisition
python acquisition/make_dataset.py --input acquisition/recordings/recording_XXXX.npz
```

输出：
- `mymat_custom/C01T.mat` — 训练集（24 trials）
- `mymat_custom/C01E.mat` — 测试集（6 trials）

### 多次录制

如果想积累更多数据，重复第 3 步，用不同 subject 编号：

```powershell
python acquisition/make_dataset.py --input recording_第二次.npz --subject 2
```

## 5. 训练模型

```powershell
conda activate eeg-moabb
python train.py C
```

## 6. 采集注意事项

### 被试要求
- 采集时**保持安静**，尽量少眨眼、少动
- 想象动作时**不要真的动手**，只在脑中想象握拳/手指运动
- 试前让被试练习几次，熟悉"运动想象"的概念

### 信号质量检查
- 开始前在 OpenBCI GUI 里观察波形，确认：
  - 所有 8 通道都有信号（没有平线）
  - 没有持续的大幅度方波（电极接触不良）
  - 闭眼时能看到 alpha 节律增强（8-12Hz，验证信号有效）
- 如果某个通道噪声大，重新涂导电膏或调整电极

### 环境要求
- 安静的房间，避免电磁干扰
- 被试坐在舒适的椅子上，面对屏幕
- 关闭不必要的电子设备

## 7. 常见问题

**Q: recorder.py 找不到 EEG 流？**
- 确认 OpenBCI GUI 的 LSL 已点 Start
- 确认 stream name 是 `obci_eeg1`

**Q: 信号全是噪声？**
- 检查参考电极（SRB2）是否接好
- 检查导电膏是否涂够
- 检查电极是否贴紧头皮

**Q: trial 数量太少，模型效果差？**
- 30 trial 确实偏少，建议多录几轮合并
- 或增加 paradigm.py 里的 `N_TRIALS_PER_CLASS`（如改为 30，则 60 trials/次）

**Q: 没有 EEG 帽怎么定位电极？**
- 用卷尺量：鼻根到枕骨粗隆的 50% 处是 Cz
- C3/C4 在 Cz 左右各 20% 处
- 参考 [10-20 系统定位法](https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG))
