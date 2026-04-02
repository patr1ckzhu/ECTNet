# ECTNet 迁移学习阶段性总结

**日期**: 2026-04-02
**作者**: Patrick
**状态**: 实验完成，结论明确

---

## 1. 背景与动机

### 1.1 问题

自采 EEG 数据量不足，模型训练效果受限：

| 被试 | 设备 | 通道 | Trials | 从零训练 Acc | 问题 |
|------|------|------|--------|-------------|------|
| Roger (S1) | 自研板 (ADS1299+STM32+HC05) | C3, C4, Cz (干电极) | 250 | 66% (seed=42) | 数据量少，结果不稳定 |
| Patrick (S2) | 同上 | C3, C4, Cz (干电极) | 60 | 50% | 严重不足，等同随机 |

### 1.2 思路

BCI Competition IV-2b 数据集与自采数据高度同构：
- 相同通道配置: C3, C4, Cz (3 通道)
- 相同任务: 左手/右手运动想象 (2 分类)
- 相同采样率: 250 Hz
- 相同模型架构: ECTNet (25K params)
- 数据充足: 9 个被试，共 6520 trials

**核心假设**: 用 2b 大数据预训练学习通用 MI 时空特征，再用自采数据微调适配个体和硬件差异。

### 1.3 文献支持

| 论文 | 方法 | 提升 |
|------|------|------|
| ConvoReleNet (2025, Frontiers Neurosci) | 8 subjects 预训练 → 微调 | BCI IV-2b: 75% → 84% (+9%) |
| Fine-Tuning Strategies (2025, IEEE) | 60 subjects 预训练 → 150 trials 微调 | 74% → 79% (+5%) |
| DARN (2023, IEEE TNSRE) | Few-shot 元学习 | 20-shot +9.3% |

---

## 2. 实验设计

### 2.1 数据

| 数据集 | 来源 | Trials | 通道 | 单位 | 用途 |
|--------|------|--------|------|------|------|
| BCI IV-2b | 竞赛数据 (湿电极, 实验室环境) | 6520 (9 subjects × T+E) | C3, C4, Cz | V (伏特) | 预训练 |
| Custom C | 自研板采集 (干电极, 蓝牙) | 250 (Roger), 200 train + 50 test | C3, C4, Cz | µV (微伏) | 微调 + 测试 |

**单位对齐**: 自采数据 µV × 1e-6 → V，与 2b 统一。Z-score 标准化后数值等价，但统一单位便于理解和调试。

### 2.2 模型配置

```
EEGTransformer (ECTNet):
  CNN: temporal_conv(1→8, 1×64) → channel_attention(SE-Net, 3ch) → spatial_conv(depthwise + 1×16) → projection
  Positional Encoding: learnable (1, 100, 16)
  Transformer: 6 blocks, heads=2, emb_size=16
  Classification: Dropout(0.5) → Linear(240→2)
  Total: ~25K params
```

B 和 C 数据使用完全相同的模型架构（3ch, 2-class）。

### 2.3 实验条件

所有条件共享:
- Optimizer: Adam (lr=0.001, betas=(0.5, 0.999))
- Epochs: 1000
- Batch size: 72
- Data augmentation: Interaug S&R (aug=3, seg=8)
- Validation ratio: 0.3 (从 200 train 中留 30% = 60 samples 做验证)
- Model selection: min validation loss
- Evaluation: 10 seeds (0-9) 平均，消除随机 split 带来的方差

| 条件 | 初始权重 | 其他 |
|------|---------|------|
| **Baseline** | 随机初始化 | 全部相同 |
| **Transfer (warm start)** | 2b 预训练权重 | 全部相同 |

**唯一变量是模型初始权重**，保证对比公平性。

### 2.4 预训练阶段

```bash
python train_transfer.py pretrain
```

- 池化 2b 全部 9 个被试的 T+E 数据 (6520 trials, 3260 per class)
- 留出 10% (652 trials) 作为 held-out test
- 训练 1000 epochs
- **结果: held-out test accuracy = 80.67%**
- Channel attention: C3(0.509), C4(0.470), Cz(0.632)
- 保存: `pretrained_B/model_pretrained.pth`

---

## 3. 实验结果

### 3.1 主要结果 (10-seed 平均)

| 条件 | Mean Acc | Std | Min | Max | Kappa |
|------|---------|-----|-----|-----|-------|
| **Baseline (随机初始化)** | **76.80%** | ±5.23% | 66% | 84% | 53.60% |
| **Transfer (预训练初始化)** | **82.20%** | ±3.03% | 76% | 86% | 64.40% |
| **提升** | **+5.40%** | — | +10% | +2% | +10.80% |

### 3.2 Seed=42 对比验证

| 代码 | Acc | Kappa | 说明 |
|------|-----|-------|------|
| 原始 train.py C | 66% | 32% | 复现成功，与之前报告一致 |
| train_transfer.py baseline | 76% | 52% | 不同 train/val split |
| train_transfer.py transfer | **80%** | **60%** | 预训练初始化 |

原始 66% 与 baseline 76% 的差异来自代码路径不同导致的 train/val split 差异（原始 train.py 在 split 前创建了额外的模型对象，消耗了 torch random state）。两者训练逻辑等价。

### 3.3 失败的策略

早期实验中尝试了多种冻结策略，均未成功：

| 策略 | 10-seed Acc | 分析 |
|------|-----------|------|
| freeze=cnn (冻结 CNN) | 53.00% ± 2.24% | CNN 特征不适配干电极数据 |
| freeze=cnn+transformer4 (冻结 CNN+4 层 Transformer) | 53.00% ± 2.05% | 同上，更严重 |
| freeze=none + 分层 LR (CNN 0.1×) | 57.60% ± 4.96% | CNN 学习率过低 |
| freeze=cnn + 20 epochs | ~56% | Epoch 太少 (仅 ~40 gradient steps) |

**失败原因分析**:

1. **Domain gap 过大**: 2b 数据来自实验室湿电极，自采数据来自自研板干电极 + 蓝牙传输。信号幅度差 14 倍 (std: 4.18µV vs 56.87µV)，噪声特征完全不同。冻结 CNN 等于锁死了不适配的特征提取器。

2. **BatchNorm 问题**: 冻结层的 BatchNorm 保持 eval 模式时，running statistics 无法适配新域数据分布。即使改用 AdaBN（保持 BN train 模式），冻结 Conv 权重仍然产生不适配的特征。

3. **梯度步数不足**: 200 train samples / 72 batch = ~2 batches/epoch。20 epochs 仅 40 个 gradient steps，不足以适配。论文推荐的 10-20 epochs 适用于每 epoch 有更多 batch 的场景。

### 3.4 成功策略的关键因素

最终成功的配置: **全层微调 + 预训练初始化 + 统一学习率 + 充分训练**

```bash
python train_transfer.py finetune \
    --pretrained pretrained_B/model_pretrained.pth \
    --freeze none --epochs 1000 --lr 0.001
```

成功原因:
1. **不冻结任何层**: 干/湿电极 domain gap 太大，所有层都需要适配
2. **预训练提供好的初始化**: 虽然特征不能直接用，但预训练权重编码了 MI 的先验知识（mu/beta 节律、空间滤波模式），引导优化到更好的解空间
3. **统一学习率 0.001**: 与从零训练一致，给 CNN 足够的学习能力适配新域
4. **1000 epochs**: 充分训练确保收敛
5. **方差减半**: 预训练初始化使不同 seed 收敛到相似的好解，鲁棒性提升

---

## 4. 关键发现

### 4.1 迁移学习有效，但方式与预期不同

- **传统策略 (冻结底层微调顶层) 无效** — 干/湿电极 domain gap 打破了"底层特征通用"的假设
- **预训练初始化 + 全层重训有效** — 预训练权重作为优化起点，而非固定的特征提取器
- 这与 ConvoReleNet 等论文的场景不同（它们在同一硬件不同被试间迁移，domain gap 小得多）

### 4.2 之前报告的 66% 被低估

- 原始 train.py seed=42 得到 66%，恰好是最差的 train/val split 之一
- 多种子平均 baseline 为 **76.80%**
- 50 个测试样本的 2% 粒度 + 30% val split 的随机性 = 巨大方差（±5%）
- **启示: 小样本实验必须报告多种子统计量**

### 4.3 信号质量差异量化

| 指标 | BCI IV-2b (湿电极) | 自采数据 (干电极) | 比值 |
|------|-------------------|-----------------|------|
| 信号 std (V) | 4.18 × 10⁻⁶ | 5.69 × 10⁻⁵ | **13.6×** |
| Channel attention 排序 | Cz > C3 > C4 | C3 > C4 > Cz | 不同 |

干电极数据噪声约为湿电极的 14 倍，且 channel attention 模式不同（干电极 Cz 权重最低，可能因为干电极在头顶的接触不如两侧稳定）。

---

## 5. 代码与复现

### 5.1 文件

| 文件 | 用途 |
|------|------|
| `train_transfer.py` | 迁移学习主脚本 (pretrain/finetune/baseline) |
| `pretrained_B/model_pretrained.pth` | 2b 预训练 checkpoint |
| `transfer_C_freeze_none/model_1.pth` | Roger 微调后的最佳模型 |

### 5.2 完整复现命令

```bash
# 在 Ubuntu 上 (ssh ubuntu)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bci-ctnet
cd /home/patrick/PycharmProjects/ECTNet

# 1. 预训练 (约 20 分钟)
python train_transfer.py pretrain

# 2. 微调 (约 30 秒/seed)
python train_transfer.py --seeds 0,1,2,3,4,5,6,7,8,9 \
    finetune --pretrained pretrained_B/model_pretrained.pth \
    --freeze none --epochs 1000 --lr 0.001

# 3. Baseline 对比
python train_transfer.py --seeds 0,1,2,3,4,5,6,7,8,9 baseline

# 4. 单次训练 (用于部署)
python train_transfer.py finetune \
    --pretrained pretrained_B/model_pretrained.pth \
    --freeze none --epochs 1000 --lr 0.001
```

### 5.3 Checkpoint 兼容性

微调后的 checkpoint 与 `realtime_inference.py` 完全兼容:
- norm_mean/norm_std 已转换回 µV 域
- 模型结构不变 (EEGTransformer, 3ch, 2-class)
- 可直接用于实时推理

---

## 6. 下一步计划

### 6.1 短期

- [ ] 用迁移学习训练的最佳模型做实时推理演示
- [ ] 继续采集 Patrick 的数据 (目标 200+ trials)，用迁移学习训练
- [ ] 采集更多 Roger 的数据 (400+ trials)，测试数据量增加后的效果

### 6.2 中期

- [ ] 端到端演示: 自研板 → 蓝牙 → 实时分类 → LEFT/RIGHT 反馈
- [ ] 考虑在线 (session-to-session) 迁移: 用前几次采集的数据预训练，新 session 快速微调

### 6.3 论文方向

本次实验为论文提供了一个清晰的 contribution point:
- **跨硬件迁移学习**: 从实验室级湿电极 BCI 到低成本干电极自研 BCI 的知识迁移
- 传统冻结策略失败，全层微调 + 预训练初始化成功
- 关键发现: domain gap 过大时，预训练权重的价值在于优化起点而非固定特征
- 50 个测试样本 × 10 seeds 的统计方法论

---

## 7. 实验日志

| 时间 | 实验 | 结果 | 备注 |
|------|------|------|------|
| Round 1 | freeze=cnn/cnn+t4, 300ep, lr=0.0001 | 50-54% | BN eval 模式阻止域适配 |
| Round 2 | AdaBN fix + 20ep | 56-60% | Gradient steps 不足 (40 steps) |
| Round 3 | 200ep, lr=0.001, 多种冻结 | 54-58% | 冻结层始终有害 |
| Round 4 | 10-seed baseline + transfer | baseline 57%, transfer 58% | 发现分层 LR bug |
| **Round 5** | **统一 LR fix, 1000ep, 10-seed** | **baseline 76.8%, transfer 82.2%** | **最终结果** |
