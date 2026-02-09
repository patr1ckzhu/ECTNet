# ECTNet

Enhanced Convolutional Transformer Network for EEG-based motor imagery classification.

## Final Year Project (毕设)

**题目**: EEG 脑机接口系统 (BCI) 与运动想象识别 (MI)

这是一个**端到端 8-channel MI-BCI 系统**，从硬件采集到实时分类推理：

### 系统架构
```
[EEG 电极] → [ADS1299 模拟前端] → SPI → [STM32 微控制器] → 蓝牙 → [PC]
                                                                        ↓
                              [分类反馈 UI] ← [ECTNet 实时推理] ← [LSL 数据流] ← [PsychoPy 实验范式]
```

### 分工
- **硬件团队**: ADS1299 + STM32 采集板设计、SPI 通信、蓝牙传输
- **Patrick (我)**: 软件全栈
  - ECTNet 模型训练（基于 BCI IV 数据集离线训练，部署时实时推理）
  - PsychoPy 实验范式（MI 任务提示）
  - Lab Streaming Layer (LSL) 数据同步
  - 实时 EEG 可视化界面
  - 实时分类反馈
  - 支持左手/右手/脚/舌头运动想象分类，也探索情绪分类

### 当前重点
1. 先在 BCI Competition IV 数据集上复现训练，验证模型性能
2. 然后做实时推理 pipeline（LSL 接收 → 预处理 → 模型推理 → 输出）
3. 最终接上自研硬件做端到端演示

### 远期目标
基于毕设成果发表论文，候选改进方向见 `docs/research_directions.md`

## Project Context

- Based on CTNet (CNN + Transformer for MI-EEG classification)
- Original paper: Zhao et al., Sci Rep 14, 20237 (2024)
- Target hardware: 自研 ADS1299+STM32 采集板 (类 OpenBCI), 8 EEG channels
- Datasets: BCI Competition IV-2a (22ch, 4-class) and IV-2b (3ch, 2-class)

## Environment

- Mac (开发) ↔ SSH → Windows PowerShell + RTX 5080 (训练)
- SSH 配置: `ssh win` (Host: 192.168.1.153, User: patrick, Key: ~/.ssh/id_ed25519_evelyn)
- Windows 代码路径: `C:\Users\Patrick\PycharmProjects\ECTNet`
- Windows conda 路径: `C:\Users\patrick\anaconda3\condabin\conda.bat`
- Conda environment: `eeg-moabb`
- Python 3.12, PyTorch 2.6+, MNE, einops, scikit-learn

### 远程训练命令
```bash
# 从 Mac 执行远程训练
ssh win "C:\Users\patrick\anaconda3\condabin\conda.bat activate eeg-moabb && cd C:\Users\Patrick\PycharmProjects\ECTNet && python train.py"
```

## Key Files

- `model.py` — CTNet model definition (ChannelAttention, PatchEmbeddingCNN, Transformer, EEGTransformer)
- `train.py` — Training logic (ExP class, main loop, hyperparameters, L1 regularization for CA)
- `utils.py` — Data loading, metrics, GradCAM visualization
- `preprocessing/` — GDF → MAT data preprocessing
- `experiments/` — Training variants (8ch, 2-class, 3-class)
- `tools/channel_selector.py` — Channel selection for reducing 22ch → 8ch
- `docs/` — Research plans and documentation

## Workflow

1. Download BCI IV-2a/2b datasets into `BCICIV_2a_gdf/` and `BCICIV_2b_gdf/`
2. Run preprocessing: `python preprocessing/preprocessing_for_2a.py`
3. Train: `python train.py` (or scripts in `experiments/`)

## Conventions

- Model architecture is defined in `model.py`, training logic in `train.py`
- Experiment variant scripts in `experiments/` still contain inline model definitions
- Data is stored as .mat files after preprocessing
- Models are saved as .pth files in `models/` directory

## Experiment Results (Seed=42, BCI IV-2a, 4-class)

### 22ch Baseline vs Channel Attention
- **22ch baseline**: 81.63% mean acc (CTNet 论文 81.33%)
- **22ch + CA (SE-Net, l1_lambda=1e-4)**: 81.29% mean acc → 基本持平，CA 在 22ch 下无显著提升
- CA 学到的通道权重符合神经科学先验（感觉运动区 C5/Cz/C6/CP3/CP4 权重高）

### 8ch 通道选择对比
- **Manual 8ch (先验知识)**: 75.50% — C3, C4, Cz, FCz, CP1, CP2, FC3, FC4
- **Learned 8ch (CA top 8)**: 73.50% — FC1, FCz, C5, Cz, C6, CP3, CP4, P2
- 结论: 手动选通道更优，CA 的 soft attention 权重反映的是 22ch 下的补充价值，不等于最优 8ch 子集
- **硬件部署推荐**: C3, C4, Cz, FCz, CP1, CP2, FC3, FC4

### Branch: `feature/channel-attention`
- model.py: 添加了 ChannelAttention (SE-Net style)，PatchEmbeddingCNN 拆分为 temporal_conv → channel_attention → spatial_conv
- train.py: L1 正则 + 训练后保存通道权重 (.npy)
- 实验脚本和结果在 Windows: `compare_9sub_baseline/`, `compare_9sub_ca/`, `compare_8ch_learned/`, `compare_8ch_manual/`

## Progress

- [x] Forked CTNet → created ECTNet repo (github.com/patr1ckzhu/ECTNet)
- [x] Refactored: split model.py + train.py
- [x] Reorganized folder structure
- [x] SSH 远程训练 pipeline 验证通过 (Mac → ssh win → conda activate → train.py)
- [x] Reproduce baseline training on Windows 5080 (9 subjects, mean acc 81.63% seed=42)
- [x] Channel Attention 实验完成 (22ch 持平, 8ch 手动选通道更优)
- [ ] 训练效率优化 (GPU 利用率仅 ~30%, 目标 80%+)
- [ ] Build real-time inference pipeline (LSL → preprocess → model → output)
- [ ] PsychoPy experiment paradigm
- [ ] Real-time EEG visualization UI
- [ ] Integration with custom ADS1299+STM32 hardware
