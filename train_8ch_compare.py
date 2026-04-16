"""
8-Channel 3-Way Comparison: Learned (CA) vs Manual (Prior Knowledge) vs 22ch Baseline

Compares three channel selection strategies for 8-channel EEG motor imagery classification:
1. Learned 8ch: Top 8 channels from channel attention (CA) weights averaged across 9 subjects
2. Manual 8ch: Prior knowledge selection based on motor imagery literature
3. 22ch baseline: Full 22-channel results from previous run (reference only)

All experiments use fixed seed=42 for reproducibility.

Author: Patrick
Date: 2025-02-09
"""

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
import numpy as np
import random
import time
import datetime
import math
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch import Tensor
from torch.backends import cudnn
from torch.autograd import Variable
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import pandas as pd

cudnn.benchmark = False
cudnn.deterministic = True

# Add project root to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import calMetrics, load_data_evaluate

# ==================== Constants ====================

CHANNEL_NAMES = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz',
                 'C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']

FIXED_SEED = 42
N_SUBJECT = 9
DATA_DIR = r'./mymat_raw/'

# Manual 8 channels (prior knowledge from motor imagery research)
MANUAL_8CH = sorted([7, 11, 9, 3, 14, 16, 1, 5])  # C3, C4, Cz, FCz, CP1, CP2, FC3, FC4


# ==================== Channel Selection ====================

def select_channels(data, channel_indices):
    """Select specified channels from 22-channel data."""
    if data.ndim == 3:
        return data[:, channel_indices, :]
    elif data.ndim == 4:
        return data[:, :, channel_indices, :]
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")


# ==================== Model Architecture (8ch, no CA) ====================

class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, number_channel=8, emb_size=40):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        return self.fc(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        return self.layernorm(self.drop(res) + x_input)


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
            ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            ), emb_size, drop_p)
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, depth=6, emb_size=40, number_channel=8,
                 f1=20, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, kernel_size=kernel_size, D=D,
                            pooling_size1=pooling_size1, pooling_size2=pooling_size2,
                            dropout_rate=dropout_rate, number_channel=number_channel,
                            emb_size=emb_size),
        )


class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)


class EEGTransformer(nn.Module):
    def __init__(self, heads=4, emb_size=40, depth=6, database_type='A',
                 eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3, eeg1_number_channel=8,
                 flatten_eeg1=600, **kwargs):
        super().__init__()
        self.number_class = 4
        self.number_channel = eeg1_number_channel
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()

        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                          f1=eeg1_f1, kernel_size=eeg1_kernel_size, D=eeg1_D,
                                          pooling_size1=eeg1_pooling_size1,
                                          pooling_size2=eeg1_pooling_size2,
                                          dropout_rate=eeg1_dropout_rate)
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1, self.number_class)

    def forward(self, x):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn + trans
        out = self.classification(self.flatten(features))
        return features, out


# ==================== Training Class ====================

class ExP8ch():
    def __init__(self, nsub, data_dir, result_name, epochs=2000, number_aug=2,
                 number_seg=8, gpus=[0], evaluate_mode='subject-dependent',
                 heads=4, emb_size=40, depth=6, dataset_type='A',
                 eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3, flatten_eeg1=600,
                 validate_ratio=0.2, learning_rate=0.001, batch_size=72,
                 selected_channels=None):

        super(ExP8ch, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.lr = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_epochs = epochs
        self.nSub = nsub
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.root = data_dir
        self.heads = heads
        self.emb_size = emb_size
        self.depth = depth
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio
        self.selected_channels = selected_channels

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.number_class = 4
        self.number_channel = 8

        self.model = EEGTransformer(
            heads=self.heads, emb_size=self.emb_size, depth=self.depth,
            database_type=self.dataset_type, eeg1_f1=eeg1_f1, eeg1_D=eeg1_D,
            eeg1_kernel_size=eeg1_kernel_size, eeg1_pooling_size1=eeg1_pooling_size1,
            eeg1_pooling_size2=eeg1_pooling_size2, eeg1_dropout_rate=eeg1_dropout_rate,
            eeg1_number_channel=self.number_channel, flatten_eeg1=flatten_eeg1,
        ).cuda()

        self.model_filename = self.result_name + '/model_{}.pth'.format(self.nSub)

    def interaug(self, timg, label):
        """Segmentation & Reconstruction data augmentation"""
        aug_data = []
        aug_label = []
        number_records_by_augmentation = self.number_augmentation * int(self.batch_size / self.number_class)
        number_segmentation_points = 1000 // self.number_seg

        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:number_records_by_augmentation])

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label - 1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        """Load and process data with channel selection"""
        (train_data, train_label, test_data, test_label) = load_data_evaluate(
            self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode
        )

        print(f"   Original data shape: {train_data.shape}")
        train_data = select_channels(train_data, self.selected_channels)
        test_data = select_channels(test_data, self.selected_channels)
        print(f"   After 8ch selection: {train_data.shape}")

        self.train_data = train_data
        self.test_data = test_data

        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)

        self.allData = train_data
        self.allLabel = train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        print(f"   Train size: {self.train_data.shape}  Test size: {self.test_data.shape}")

        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)

        self.testData = test_data
        self.testLabel = test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        """Training loop"""
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        best_epoch = 0
        num = 0
        min_loss = 100
        result_process = []

        for e in range(self.n_epochs):
            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
            epoch_process = {}
            epoch_process['epoch'] = e
            self.model.train()
            outputs_list = []
            label_list = []
            val_data_list = []
            val_label_list = []

            for i, (img, label) in enumerate(self.dataloader):
                number_sample = img.shape[0]
                number_validate = int(self.validate_ratio * number_sample)

                train_data = img[:-number_validate]
                train_label = label[:-number_validate]

                val_data_list.append(img[-number_validate:])
                val_label_list.append(label[-number_validate:])

                img = Variable(train_data.type(self.Tensor))
                label = Variable(train_label.type(self.LongTensor))

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                features, outputs = self.model(img)
                outputs_list.append(outputs)
                label_list.append(label)

                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            del img
            torch.cuda.empty_cache()

            if (e + 1) % 1 == 0:
                self.model.eval()
                val_data = torch.cat(val_data_list).cuda()
                val_label = torch.cat(val_label_list).cuda()
                val_data = val_data.type(self.Tensor)
                val_label = val_label.type(self.LongTensor)

                val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
                self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
                outputs_list = []

                with torch.no_grad():
                    for i, (img, _) in enumerate(self.val_dataloader):
                        img = img.type(self.Tensor).cuda()
                        _, Cls = self.model(img)
                        outputs_list.append(Cls)
                        del img, Cls
                        torch.cuda.empty_cache()

                Cls = torch.cat(outputs_list)

                val_loss = self.criterion_cls(Cls, val_label)
                val_pred = torch.max(Cls, 1)[1]
                val_acc = float((val_pred == val_label).cpu().numpy().astype(int).sum()) / float(val_label.size(0))

                epoch_process['val_acc'] = val_acc
                epoch_process['val_loss'] = val_loss.detach().cpu().numpy()

                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                epoch_process['train_acc'] = train_acc
                epoch_process['train_loss'] = loss.detach().cpu().numpy()

                num = num + 1

                if min_loss > val_loss:
                    min_loss = val_loss
                    best_epoch = e
                    epoch_process['epoch'] = e
                    torch.save(self.model, self.model_filename)
                    print("{}_{} train_acc: {:.4f} train_loss: {:.6f}\tval_acc: {:.6f} val_loss: {:.7f}".format(
                        self.nSub, epoch_process['epoch'], epoch_process['train_acc'],
                        epoch_process['train_loss'], epoch_process['val_acc'], epoch_process['val_loss']))

            result_process.append(epoch_process)

            del label, val_data, val_label
            torch.cuda.empty_cache()

        # Test
        self.model.eval()
        self.model = torch.load(self.model_filename, weights_only=False).cuda()
        outputs_list = []

        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_dataloader):
                img_test = Variable(img.type(self.Tensor)).cuda()
                features, outputs = self.model(img_test)
                outputs_list.append(outputs)

        outputs = torch.cat(outputs_list)
        y_pred = torch.max(outputs, 1)[1]

        test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
        print("epoch: ", best_epoch, '\tThe test accuracy is:', test_acc)

        df_process = pd.DataFrame(result_process)
        return test_acc, test_label, y_pred, df_process, best_epoch


# ==================== Main ====================

if __name__ == "__main__":

    # Load learned channel indices from CA weights
    print("=" * 70)
    print("  STEP 1: Loading channel attention weights from 9-subject CA run")
    print("=" * 70)

    ca_weights = []
    for sub in range(1, 10):
        w = np.load(f'compare_9sub_ca/channel_attention_sub{sub}.npy')
        ca_weights.append(w)
        print(f"  Sub {sub}: top3 = {CHANNEL_NAMES[np.argsort(w)[-1]]}, "
              f"{CHANNEL_NAMES[np.argsort(w)[-2]]}, {CHANNEL_NAMES[np.argsort(w)[-3]]}")

    avg_weights = np.mean(ca_weights, axis=0)
    LEARNED_8CH = sorted(np.argsort(avg_weights)[-8:][::-1].tolist())
    LEARNED_8CH_NAMES = [CHANNEL_NAMES[i] for i in LEARNED_8CH]

    print(f"\n  Average CA weights across 9 subjects:")
    for i, (name, w) in enumerate(zip(CHANNEL_NAMES, avg_weights)):
        marker = " <-- SELECTED" if i in LEARNED_8CH else ""
        print(f"    {i:2d} {name:>4s}: {w:.4f}{marker}")

    print(f"\n  Learned 8ch: indices={LEARNED_8CH}, names={LEARNED_8CH_NAMES}")
    print(f"  Manual  8ch: indices={MANUAL_8CH}, names={[CHANNEL_NAMES[i] for i in MANUAL_8CH]}")
    overlap = sorted(set(LEARNED_8CH) & set(MANUAL_8CH))
    print(f"  Overlap ({len(overlap)} channels): indices={overlap}, names={[CHANNEL_NAMES[i] for i in overlap]}")
    print(f"  Only in learned: {sorted(set(LEARNED_8CH) - set(MANUAL_8CH))} = {[CHANNEL_NAMES[i] for i in sorted(set(LEARNED_8CH) - set(MANUAL_8CH))]}")
    print(f"  Only in manual:  {sorted(set(MANUAL_8CH) - set(LEARNED_8CH))} = {[CHANNEL_NAMES[i] for i in sorted(set(MANUAL_8CH) - set(LEARNED_8CH))]}")

    # Hyperparameters (same as train_8ch.py defaults)
    EPOCHS = 1000
    N_AUG = 3
    N_SEG = 8
    EMB_DIM = 16
    HEADS = 2
    DEPTH = 6
    VALIDATE_RATIO = 0.3

    EEGNet1_F1 = 8
    EEGNet1_KERNEL_SIZE = 64
    EEGNet1_D = 2
    EEGNet1_POOL_SIZE1 = 8
    EEGNet1_POOL_SIZE2 = 8
    EEGNet1_DROPOUT_RATE = 0.5
    FLATTEN_EEGNet1 = 240

    def run_8ch_experiment(label, result_dir, channel_indices):
        """Run all 9 subjects with given 8-channel selection, fixed seed."""
        print(f"\n{'=' * 70}")
        print(f"  RUNNING: {label}")
        print(f"  Channels: {channel_indices} = {[CHANNEL_NAMES[i] for i in channel_indices]}")
        print(f"  Output dir: {result_dir}")
        print(f"  Seed: {FIXED_SEED}, Epochs: {EPOCHS}")
        print(f"{'=' * 70}")

        os.makedirs(result_dir, exist_ok=True)
        results = []

        for sub in range(1, N_SUBJECT + 1):
            # Fixed seed for each subject
            random.seed(FIXED_SEED)
            np.random.seed(FIXED_SEED)
            torch.manual_seed(FIXED_SEED)
            torch.cuda.manual_seed(FIXED_SEED)
            torch.cuda.manual_seed_all(FIXED_SEED)

            print(f"\n--- {label} - Subject {sub}/9 ---")
            start = time.time()

            exp = ExP8ch(
                sub, DATA_DIR, result_dir, EPOCHS, N_AUG, N_SEG, [0],
                evaluate_mode='LOSO-No', heads=HEADS, emb_size=EMB_DIM, depth=DEPTH,
                dataset_type='A',
                eeg1_f1=EEGNet1_F1, eeg1_kernel_size=EEGNet1_KERNEL_SIZE, eeg1_D=EEGNet1_D,
                eeg1_pooling_size1=EEGNet1_POOL_SIZE1, eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
                eeg1_dropout_rate=EEGNet1_DROPOUT_RATE, flatten_eeg1=FLATTEN_EEGNet1,
                validate_ratio=VALIDATE_RATIO,
                selected_channels=channel_indices
            )

            testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
            elapsed = time.time() - start

            acc, prec, rec, f1, kappa = calMetrics(
                Y_true.cpu().numpy().astype(int),
                Y_pred.cpu().numpy().astype(int)
            )
            results.append({
                'sub': sub, 'acc': acc * 100, 'kappa': kappa * 100,
                'precision': prec * 100, 'recall': rec * 100, 'f1': f1 * 100,
                'epoch': best_epoch, 'time': elapsed
            })
            print(f"  >> {label} S{sub}: acc={acc*100:.2f}%, kappa={kappa*100:.2f}%, "
                  f"best_epoch={best_epoch}, time={elapsed:.1f}s")

            del exp
            torch.cuda.empty_cache()

        return results

    # ==================== Run experiments ====================

    total_start = time.time()

    # Condition 1: Learned 8ch
    results_learned = run_8ch_experiment(
        "LEARNED 8CH (from CA weights)",
        "compare_8ch_learned",
        LEARNED_8CH
    )

    # Condition 2: Manual 8ch
    results_manual = run_8ch_experiment(
        "MANUAL 8CH (prior knowledge)",
        "compare_8ch_manual",
        MANUAL_8CH
    )

    total_elapsed = time.time() - total_start

    # ==================== Print Results ====================

    print(f"\n\n{'#' * 80}")
    print(f"#{'':^78s}#")
    print(f"#{'3-WAY COMPARISON RESULTS':^78s}#")
    print(f"#{'Fixed Seed = 42':^78s}#")
    print(f"#{'':^78s}#")
    print(f"{'#' * 80}")

    print(f"\n  Learned 8ch: {LEARNED_8CH} = {LEARNED_8CH_NAMES}")
    print(f"  Manual  8ch: {MANUAL_8CH} = {[CHANNEL_NAMES[i] for i in MANUAL_8CH]}")
    print(f"  22ch baseline: all 22 channels (from compare_9sub_baseline)")
    print(f"  Overlap (learned vs manual): {len(overlap)} channels = {[CHANNEL_NAMES[i] for i in overlap]}")

    # 22ch baseline results from previous run (seed=42)
    # These are hardcoded from the known previous run
    baseline_22ch = {
        'note': 'From compare_9sub_baseline (seed=42), mean=81.63%',
        'label': '22ch Baseline'
    }

    print(f"\n  {'Sub':<5} | {'22ch Base':>11} | {'Learned 8ch':>13} | {'Manual 8ch':>12} | {'L vs 22ch':>10} | {'M vs 22ch':>10} | {'L vs M':>8}")
    print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*13}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    accs_l = [r['acc'] for r in results_learned]
    accs_m = [r['acc'] for r in results_manual]
    kaps_l = [r['kappa'] for r in results_learned]
    kaps_m = [r['kappa'] for r in results_manual]

    for rl, rm in zip(results_learned, results_manual):
        sub = rl['sub']
        delta_lm = rl['acc'] - rm['acc']
        print(f"  S{sub:<3} | {'N/A':>10}  | {rl['acc']:>12.2f}% | {rm['acc']:>11.2f}% | {'N/A':>10} | {'N/A':>10} | {delta_lm:>+7.2f}%")

    print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*13}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    mean_l = np.mean(accs_l)
    mean_m = np.mean(accs_m)
    std_l = np.std(accs_l)
    std_m = np.std(accs_m)
    print(f"  {'Mean':<5} | {'81.63%':>11} | {mean_l:>12.2f}% | {mean_m:>11.2f}% | {mean_l-81.63:>+9.2f}% | {mean_m-81.63:>+9.2f}% | {mean_l-mean_m:>+7.2f}%")
    print(f"  {'Std':<5} | {'---':>11} | {std_l:>12.2f}% | {std_m:>11.2f}% |")

    print(f"\n  Kappa Scores:")
    print(f"  {'Sub':<5} | {'Learned':>10} | {'Manual':>10} | {'Delta':>8}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for rl, rm in zip(results_learned, results_manual):
        print(f"  S{rl['sub']:<3} | {rl['kappa']:>9.2f}% | {rm['kappa']:>9.2f}% | {rl['kappa']-rm['kappa']:>+7.2f}%")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'Mean':<5} | {np.mean(kaps_l):>9.2f}% | {np.mean(kaps_m):>9.2f}% | {np.mean(kaps_l)-np.mean(kaps_m):>+7.2f}%")
    print(f"  {'Std':<5} | {np.std(kaps_l):>9.2f}% | {np.std(kaps_m):>9.2f}% |")

    print(f"\n  Training Time:")
    print(f"  Learned 8ch total: {sum(r['time'] for r in results_learned):.0f}s ({sum(r['time'] for r in results_learned)/60:.1f} min)")
    print(f"  Manual  8ch total: {sum(r['time'] for r in results_manual):.0f}s ({sum(r['time'] for r in results_manual)/60:.1f} min)")
    print(f"  Combined total:    {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    print(f"\n{'#' * 80}")
    print(f"  DONE. Results saved to compare_8ch_learned/ and compare_8ch_manual/")
    print(f"{'#' * 80}")
