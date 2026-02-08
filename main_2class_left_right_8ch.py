"""
CTNet: 8-Channel 2-Class Version (Left Hand vs Right Hand)

This script trains CTNet model using 8 selected channels for binary classification
(left hand vs right hand) from the BCI IV-2a dataset.

Based on the original CTNet paper:
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based
motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7

Modified by: Patrick
Date: 2025-10-19
"""

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import pandas as pd
import random
import datetime
import time

from pandas import ExcelWriter
from torchsummary import summary
import torch
from torch.backends import cudnn
from utils import calMetrics
import scipy.io as sio
import math
import warnings
warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True

import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
from torch.autograd import Variable


# ==================== Channel Selection ====================
# 选择的8个通道 (从channel_selector.py获取)
# 使用方法2 (互信息) 的结果 - 2025-10-19

# 方法2 (互信息 - 推荐): 基于2592个样本的数据驱动选择
SELECTED_CHANNEL_INDICES = [20, 21, 16, 18, 13, 14, 11, 19]
SELECTED_CHANNEL_NAMES = ['P2', 'POz', 'CP2', 'P1', 'CP3', 'CP1', 'C4', 'Pz']

# 备选方案:
# 方法1 (先验知识): [7, 11, 9, 3, 14, 16, 1, 5] - ['C3', 'C4', 'Cz', 'FCz', 'CP1', 'CP2', 'FC3', 'FC4']
# 方法3 (RFE模型): [17, 0, 20, 21, 18, 15, 13, 9] - ['CP4', 'Fz', 'P2', 'POz', 'P1', 'CPz', 'CP3', 'Cz']


def select_channels(data, channel_indices):
    """
    从22通道数据中选择指定的通道

    参数:
        data: (n_trials, n_channels, n_timepoints) 或 (n_trials, 1, n_channels, n_timepoints)
        channel_indices: 要选择的通道索引列表

    返回:
        selected_data: 选中通道的数据
    """
    if data.ndim == 3:
        # (n_trials, 22, 1000) -> (n_trials, 8, 1000)
        return data[:, channel_indices, :]
    elif data.ndim == 4:
        # (n_trials, 1, 22, 1000) -> (n_trials, 1, 8, 1000)
        return data[:, :, channel_indices, :]
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")


def load_data_2class(dir_path, dataset_type, n_sub):
    """
    加载2分类数据(只保留左手和右手)

    参数:
        dir_path: 数据目录路径
        dataset_type: 'A' 或 'B'
        n_sub: 受试者编号 (1-9)

    返回:
        train_data: (N, 22, 1000) - 只包含左右手的训练数据
        train_label: (N, 1) - 标签为0(左手)或1(右手)
        test_data: (N, 22, 1000)
        test_label: (N, 1)
    """

    # 加载原始数据
    train_mat = sio.loadmat(dir_path + dataset_type + '0' + str(n_sub) + 'T.mat')
    test_mat = sio.loadmat(dir_path + dataset_type + '0' + str(n_sub) + 'E.mat')

    train_data_full = train_mat['data']  # (288, 22, 1000)
    train_label_full = train_mat['label']  # (288, 1)
    test_data_full = test_mat['data']
    test_label_full = test_mat['label']

    # 只保留左手(class 1)和右手(class 2)
    train_mask = np.isin(train_label_full, [1, 2]).flatten()
    test_mask = np.isin(test_label_full, [1, 2]).flatten()

    train_data = train_data_full[train_mask]
    train_label = train_label_full[train_mask]
    test_data = test_data_full[test_mask]
    test_label = test_label_full[test_mask]

    # 重新映射标签: 1->0, 2->1 (方便pytorch交叉熵)
    train_label = train_label - 1  # 1->0, 2->1
    test_label = test_label - 1

    print(f"  原始数据: 训练集 {train_data_full.shape[0]} trials, 测试集 {test_data_full.shape[0]} trials")
    print(f"  2分类数据: 训练集 {train_data.shape[0]} trials, 测试集 {test_data.shape[0]} trials")
    print(f"  左手(0): 训练集 {(train_label==0).sum()}, 测试集 {(test_label==0).sum()}")
    print(f"  右手(1): 训练集 {(train_label==1).sum()}, 测试集 {(test_label==1).sum()}")

    return train_data, train_label, test_data, test_label


# ==================== Model Architecture ====================

class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, number_channel=8, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            # channel depth-wise conv (8通道)
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 2
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
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
        out = self.fc(x)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res)+x_input)
        return out


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
    def __init__(self, heads=4, emb_size=40, depth=6, n_classes=2,
                 eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3, eeg1_number_channel=8,
                 flatten_eeg1=600, **kwargs):
        super().__init__()
        self.number_class = n_classes  # 2分类
        self.number_channel = eeg1_number_channel  # 使用8通道
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

class ExP():
    def __init__(self, nsub, data_dir, result_name, epochs=2000, number_aug=2,
                 number_seg=8, gpus=[0],
                 heads=4, emb_size=40, depth=6, dataset_type='A',
                 eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3, flatten_eeg1=600,
                 validate_ratio=0.2, learning_rate=0.001, batch_size=72,
                 selected_channels=None):

        super(ExP, self).__init__()
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
        self.validate_ratio = validate_ratio
        self.selected_channels = selected_channels if selected_channels else SELECTED_CHANNEL_INDICES

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        # 固定为2类，8通道
        self.number_class = 2
        self.number_channel = 8

        self.model = EEGTransformer(
            heads=self.heads, emb_size=self.emb_size, depth=self.depth,
            n_classes=self.number_class, eeg1_f1=eeg1_f1, eeg1_D=eeg1_D,
            eeg1_kernel_size=eeg1_kernel_size, eeg1_pooling_size1=eeg1_pooling_size1,
            eeg1_pooling_size2=eeg1_pooling_size2, eeg1_dropout_rate=eeg1_dropout_rate,
            eeg1_number_channel=self.number_channel, flatten_eeg1=flatten_eeg1,
        ).cuda()

        self.model = self.model.cuda()
        self.model_filename = self.result_name + '/model_{}.pth'.format(self.nSub)

    def interaug(self, timg, label):
        """Segmentation & Reconstruction data augmentation - 适配2分类"""
        aug_data = []
        aug_label = []
        # 2分类: 每类batch_size/2个样本
        number_records_by_augmentation = self.number_augmentation * int(self.batch_size / self.number_class)
        number_segmentation_points = 1000 // self.number_seg

        for clsAug in range(self.number_class):
            # 标签已经是0和1，直接使用
            cls_idx = np.where(label == clsAug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]

            aug_data.append(tmp_aug_data)
            # 创建新的标签数组，而不是切片原有标签
            aug_label.append(np.full(number_records_by_augmentation, clsAug, dtype=np.int64))

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()  # 标签已经是0和1，不需要-1
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        """Load and process 2-class data with channel selection"""
        # 加载2分类数据（标签已经是0和1）
        (train_data, train_label, test_data, test_label) = load_data_2class(
            self.root, self.dataset_type, self.nSub
        )

        # 通道选择 (22 -> 8)
        print(f"   原始数据形状: {train_data.shape}")
        train_data = select_channels(train_data, self.selected_channels)
        test_data = select_channels(test_data, self.selected_channels)
        print(f"   选择8通道后: {train_data.shape}")

        # 保存到self (用于后续打印)
        self.train_data = train_data
        self.test_data = test_data

        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)

        self.allData = train_data
        self.allLabel = train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        print('-' * 20, "train size：", self.train_data.shape, "test size：", self.test_data.shape)

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
        label = torch.from_numpy(label)  # 标签已经是0和1
        dataset = torch.utils.data.TensorDataset(img, label)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)  # 标签已经是0和1
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
                val_pred = torch.max(outputs, 1)[1]
                outputs_list.append(outputs)

        outputs = torch.cat(outputs_list)
        y_pred = torch.max(outputs, 1)[1]

        test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

        print("epoch: ", best_epoch, '\tThe test accuracy is:', test_acc)

        df_process = pd.DataFrame(result_process)

        return test_acc, test_label, y_pred, df_process, best_epoch


# ==================== Main Function ====================

def main(dirs, heads=8, emb_size=48, depth=3,
         dataset_type='A', eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
         eeg1_pooling_size1=8, eeg1_pooling_size2=8, eeg1_dropout_rate=0.3,
         flatten_eeg1=600, validate_ratio=0.2, selected_channels=None):

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    result_write_metric = ExcelWriter(dirs + "/result_metric.xlsx")

    result_metric_dict = {}
    y_true_pred_dict = {}

    process_write = ExcelWriter(dirs + "/process_train.xlsx")
    pred_true_write = ExcelWriter(dirs + "/pred_true.xlsx")
    subjects_result = []
    best_epochs = []

    print("\n" + "="*60)
    print(f"8-Channel 2-Class CTNet Training (Left vs Right)")
    print(f"Selected Channels: {selected_channels}")
    print(f"Channel Names: {SELECTED_CHANNEL_NAMES}")
    print("="*60 + "\n")

    for i in range(N_SUBJECT):

        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2024)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i + 1))
        exp = ExP(i + 1, DATA_DIR, dirs, EPOCHS, N_AUG, N_SEG, gpus,
                  heads=heads, emb_size=emb_size,
                  depth=depth, dataset_type=dataset_type, eeg1_f1=eeg1_f1,
                  eeg1_kernel_size=eeg1_kernel_size, eeg1_D=eeg1_D,
                  eeg1_pooling_size1=eeg1_pooling_size1,
                  eeg1_pooling_size2=eeg1_pooling_size2,
                  eeg1_dropout_rate=eeg1_dropout_rate,
                  flatten_eeg1=flatten_eeg1, validate_ratio=validate_ratio,
                  selected_channels=selected_channels)

        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i + 1))
        y_true_pred_dict[i] = df_pred_true

        accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {'accuray': accuracy * 100, 'precision': precison * 100,
                          'recall': recall * 100, 'f1': f1 * 100, 'kappa': kappa * 100}
        subjects_result.append(subject_result)
        df_process.to_excel(process_write, sheet_name=str(i + 1))
        best_epochs.append(best_epoch)

        print(' THE BEST ACCURACY IS ' + str(testAcc) + "\tkappa is " + str(kappa))

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))

        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

        df_result = pd.DataFrame(subjects_result)

    process_write.close()
    pred_true_write.close()

    print('**The average Best accuracy is: ' + str(df_result['accuray'].mean()) +
          " kappa is: " + str(df_result['kappa'].mean()) + "\n")
    print("best epochs: ", best_epochs)

    result_metric_dict = df_result

    mean = df_result.mean(axis=0)
    mean.name = 'mean'
    std = df_result.std(axis=0)
    std.name = 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])

    df_result.to_excel(result_write_metric, index=False)
    print('-' * 9, ' all result ', '-' * 9)
    print(df_result)

    print("*" * 40)

    result_write_metric.close()

    return result_metric_dict


if __name__ == "__main__":
    # ==================== Configuration ====================
    DATA_DIR = r'./mymat_raw/'

    N_SUBJECT = 9
    N_AUG = 3
    N_SEG = 8

    EPOCHS = 1000
    EMB_DIM = 16
    HEADS = 2
    DEPTH = 6
    TYPE = 'A'
    validate_ratio = 0.3

    EEGNet1_F1 = 8
    EEGNet1_KERNEL_SIZE = 64
    EEGNet1_D = 2
    EEGNet1_POOL_SIZE1 = 8
    EEGNet1_POOL_SIZE2 = 8
    FLATTEN_EEGNet1 = 240

    EEGNet1_DROPOUT_RATE = 0.5

    # 使用选定的8个通道
    SELECTED_CHANNELS = SELECTED_CHANNEL_INDICES
    RESULT_NAME = "{}_2class_8channels_heads_{}_depth_{}".format(TYPE, HEADS, DEPTH)

    # 显示模型架构
    sModel = EEGTransformer(
        heads=HEADS, emb_size=EMB_DIM, depth=DEPTH, n_classes=2,
        eeg1_f1=EEGNet1_F1, eeg1_D=EEGNet1_D, eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
        eeg1_pooling_size1=EEGNet1_POOL_SIZE1, eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
        eeg1_dropout_rate=EEGNet1_DROPOUT_RATE, eeg1_number_channel=8,
        flatten_eeg1=FLATTEN_EEGNet1,
    ).cuda()

    print("\n" + "="*60)
    print("Model Architecture (8-Channel 2-Class Version)")
    print("="*60)
    summary(sModel, (1, 8, 1000))  # 8通道输入

    print(time.asctime(time.localtime(time.time())))

    result = main(RESULT_NAME, heads=HEADS,
                  emb_size=EMB_DIM, depth=DEPTH, dataset_type=TYPE,
                  eeg1_f1=EEGNet1_F1, eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
                  eeg1_D=EEGNet1_D, eeg1_pooling_size1=EEGNet1_POOL_SIZE1,
                  eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
                  eeg1_dropout_rate=EEGNet1_DROPOUT_RATE,
                  flatten_eeg1=FLATTEN_EEGNet1, validate_ratio=validate_ratio,
                  selected_channels=SELECTED_CHANNELS)

    print(time.asctime(time.localtime(time.time())))
