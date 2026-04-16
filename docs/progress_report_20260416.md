# 进度汇报 — 2026.04.16

## 本次完成

### 1. 多被试 MI 数据采集（新增 Mark + Bowen）

在已有 Roger（250 trials）、Patrick（60 trials）基础上新增两位被试，使用自研板 ADS1299+STM32+HC05 3ch（C3/C4/Cz）干电极 + 蓝牙传输。

### 2. SSH rog 配置

配置本 Mac 直接 ssh rog（免密登录），允许远程诊断 Windows 端采集问题。同步更新 M1 Max 的 IP（192.168.1.111 → 192.168.1.126）。

### 3. Kinesthetic MI 技巧教学

为 Mark 区分 visual MI（看画面）和 kinesthetic MI（感受肌肉），Mark Run 6 后 C4 ERD 方向首次正确（−3.8%）。

---

## Subject ID 约定（mymat_custom 里的 C0x 槽位）

| Slot | 被试 | 当前状态 | 最佳结果 |
|---|---|---|---|
| **C01** | **Roger** | 250 trials, 已部署 | baseline 76.8% / transfer+aug **85.4%** ⭐ |
| **C02** | **Patrick** | 60 trials, 数据不足 | 50% 随机 |
| **C03** | **Bowen** | 60 trials 稳定 + 90 trials 右手漂移 | baseline **67.5%** (Run 1+3) |
| **C04** | **Mark** | 90 trials (visual) + 30 trials (kinesthetic) | baseline 54.4% |

---

## Recordings 归档（ROG 笔记本）

路径：`C:\Users\zhupx\OneDrive\Desktop\PyCharmProjects\ECTNet\acquisition\recordings\`

### Roger（Subject 1）

#### OpenBCI Cyton 早期（3ch, 80 trials, 废弃）
- `recording_20260226_222422.npz`
- `recording_20260226_225207.npz`
- `recording_20260226_231934.npz`

#### 自研板（250 trials 合计，已部署）
| 文件 | trials | C4 ERD |
|---|---|---|
| `recording_20260401_203335.npz` | pilot | - |
| `recording_20260401_214657.npz` | pilot | - |
| `recording_20260401_222853.npz` | pilot | - |
| `recording_20260401_230223.npz` | 60 | +10% |
| `recording_20260401_231558.npz` | 30 | +13% |
| `recording_20260402_011153.npz` | 50 | +15% |
| `recording_20260402_012204.npz` | 50 | **−38%** ⭐⭐⭐ |
| `recording_20260402_014339.npz` | 60 | +20% |

### Patrick（Subject 2）
| 文件 | trials |
|---|---|
| `recording_20260402_000323.npz` | 60 |

### Mark（Subject 4，2026-04-16）
| 文件 | trials | C3 ERD | C4 ERD | 状态 |
|---|---|---|---|---|
| `recording_20260416_175324.npz` | 30 | +10% | +6% | ❌ 胳膊想象，丢弃 |
| `recording_20260416_182007.npz` | 0 | - | - | ❌ 空 marker |
| `recording_20260416_182817.npz` | 30 | +2% | **−8.4%** | ⚠️ visual, C4 方向对 |
| `recording_20260416_184039.npz` | 30 | −5% | −2.6% | ⚠️ visual |
| `recording_20260416_184936.npz` | 30 | −2% | −4.9% | ⚠️ visual |
| `recording_20260416_213415.npz` | 30 | +5% | +7% ❌ | ❌ 信号干净但 C4 反向 |
| `recording_20260416_214342.npz` | 30 | +4% | **−3.8%** | ⭐ kinesthetic breakthrough |

**当前 `C04T/E.mat`** = Run 2+3+4 (90 trials) → baseline 54.4% ± 4.2%

### Bowen（Subject 3，2026-04-16 晚上）
| 文件 | trials | C3 ERD | C4 ERD | 状态 |
|---|---|---|---|---|
| `recording_20260416_190227.npz` | 30 | −8% | **−14.2%** ⭐ | ✅ Run 1 左手主导 |
| `recording_20260416_191111.npz` | 30 | +33% | +0.6% | ❌ C3 饱和，丢弃 |
| `recording_20260416_191500.npz` | 0 | - | - | ❌ 空 marker |
| `recording_20260416_192030.npz` | 30 | +13% | **−19.0%** ⭐⭐ | ✅ Run 3 双侧 ERD |
| `recording_20260416_220529.npz` | 30 | −3% | −3.6% | ⚠️ Run 4 C3 坏 |
| `recording_20260416_221410.npz` | 30 | **+44%** | +4% | ⚠️ Run 5 MI 转右手, 1 outlier |
| `recording_20260416_221849.npz` | 0 | - | - | ❌ 空 marker |
| `recording_20260416_222428.npz` | 30 | +16% | +8% | ⚠️ Run 6 右手 only |
| `recording_20260416_223037.npz` | 30 | **+22%** | +9% | ⚠️ Run 7 右手 only, C4 std 9.56µV |
| `recording_20260416_230148.npz` | 30 | +3% | +11% | ❌ Run 8 电极帽移位全饱和 |

---

## Mac / Ubuntu 本地副本

### Ubuntu（`/home/patrick/PycharmProjects/ECTNet/acquisition/`）

| 目录 | 内容 |
|---|---|
| `recordings/` | Roger (20260226, 20260401, 20260402_01*) + Patrick (20260402_000323) |
| `recordings_bowen/` | Bowen Run 1,3,5,6,7，命名 `bowen_run{N}.npz` |
| `recordings_mark/` | Mark Run 2,3,4，命名 `mark_run{N}.npz` |

### 仅在 ROG（未同步到 Ubuntu）
- Mark Run 5 (20260416_213415) — visual, 可能无用
- Mark Run 6 (20260416_214342) — **kinesthetic breakthrough**，值得保留
- Bowen 废弃的 Run 2/4/8 和 3 个空 marker 文件

---

## Bowen 训练结果对比

| 数据组合 | trials | baseline | transfer | transfer+aug |
|---|---|---|---|---|
| Run 1+3 | 60 | **67.5% ± 7.9%** ⭐ | 68.3% ± 6.2% | 45.8% ± 5.6% ❌ |
| Run 5+6+7 | 90 | 48.3% ± 6.1% ❌ | - | - |
| Run 1+3+5+6+7 | 150 | 56.7% ± 6.7% | 59.3% ± 3.3% | 48.7% ± 5.8% |

**关键结论**：
- Run 5/6/7 单独训练 48.3% < 随机 → 左手 MI 缺失严重
- 合并 150 trials 反而比 60 trials 差 10% → MI 策略漂移污染训练集
- 小数据集下 online-aug 有害（Roger 250 trials 时才有用）

---

## 已发现问题

1. **`C03T/E.mat` 当前是 subset 实验残留**（Run 5+6+7），不是 Bowen 最好数据
2. Mark 后续 Run 5/Run 6 (kinesthetic) 未同步到 Ubuntu
3. Mark 疑似 BCI illiterate，需要额外 MI 训练
4. Bowen 连续 8 轮后左手 MI 丢失，需要下次采集前专门训练左手

---

## 下一步

1. 重建 `C03T/E.mat` = Bowen Run 1+3（60 trials）作为稳定基线
2. 下次 Bowen 采集前让他专门练左手 kinesthetic MI 5 分钟
3. Bowen 目标 200+ trials 匹配 Roger 成功条件
4. Mark Run 6 (kinesthetic) 单独保留用于后续 BCI illiteracy 研究
5. 将本 manifest 提交 git 方便未来会话查询
