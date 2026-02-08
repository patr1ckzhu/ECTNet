"""
CTNet for 2-Class Motor Imagery Classification (Left Hand vs Right Hand)
针对BCI Competition IV-2a数据集的二分类训练脚本
作者: Patrick
日期: 2025-10-19

关键修改:
1. 只使用左手(class 1)和右手(class 2)的数据
2. 输出类别数改为2
3. 数据增强后样本数增加
4. 预期准确率提升5-8%
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
from utils import calculatePerClass
from utils import numberClassChannel
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

from utils import numberClassChannel
import scipy.io as sio

import numpy as np
import pandas as pd
from torch.autograd import Variable


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D*f1
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


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=2, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
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


class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)


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


class EEGTransformer(nn.Module):
    def __init__(self, heads=2, emb_size=16, depth=6, n_classes=2, number_channel=22,
                 f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, flatten_eeg1=240, **kwargs):
        super().__init__()
        self.number_class = n_classes
        self.number_channel = number_channel
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()

        self.cnn = PatchEmbeddingCNN(f1=f1, kernel_size=kernel_size, D=D,
                                    pooling_size1=pooling_size1, pooling_size2=pooling_size2,
                                    dropout_rate=dropout_rate, number_channel=number_channel,
                                    emb_size=emb_size)
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(flatten_eeg1, n_classes)

    def forward(self, x):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn + trans
        out = self.classification(self.flatten(features))
        return features, out


def load_data_2class(dir_path, dataset_type, n_sub):
    """
    加载2分类数据(只保留左手和右手)

    参数:
        dir_path: 数据目录路径
        dataset_type: 'A' 或 'B'
        n_sub: 受试者编号 (1-9)

    返回:
        train_data: (N, 22, 1000) - 只包含左右手的训练数据
        train_label: (N, 1) - 标签为1(左手)或2(右手)
        test_data: (N, 22, 1000)
        test_label: (N, 1)
    """

    # 加载原始数据
    train_mat = sio.loadmat(dir_path + dataset_type + '0' + str(n_sub) + 'T.mat')
    test_mat = sio.loadmat(dir_path + dataset_type + '0' + str(n_sub) + 'E.mat')

    train_data_full = train_mat['data']  # (288, 1000, 22)
    train_label_full = train_mat['label']  # (288, 1)
    test_data_full = test_mat['data']
    test_label_full = test_mat['label']

    # 🔥 只保留左手(class 1)和右手(class 2)
    # 左手: label==1, 右手: label==2
    train_mask = np.isin(train_label_full, [1, 2]).flatten()
    test_mask = np.isin(test_label_full, [1, 2]).flatten()

    train_data = train_data_full[train_mask]
    train_label = train_label_full[train_mask]
    test_data = test_data_full[test_mask]
    test_label = test_label_full[test_mask]

    print(f"  DEBUG - 过滤后维度: {train_data.shape}")

    # 数据已经是(N, 22, 1000)格式,不需要转置!
    # train_data = np.transpose(train_data, (0, 2, 1))  # 注释掉
    # test_data = np.transpose(test_data, (0, 2, 1))

    print(f"  DEBUG - 最终维度: {train_data.shape}")

    # 重新映射标签: 1->0, 2->1 (方便pytorch交叉熵)
    train_label = train_label - 1  # 1->0, 2->1
    test_label = test_label - 1

    print(f"  原始数据: 训练集 {train_data_full.shape[0]} trials, 测试集 {test_data_full.shape[0]} trials")
    print(f"  2分类数据: 训练集 {train_data.shape[0]} trials, 测试集 {test_data.shape[0]} trials")
    print(f"  训练数据维度: {train_data.shape} (应该是 N, 22, 1000)")
    print(f"  左手(0): 训练集 {(train_label==0).sum()}, 测试集 {(test_label==0).sum()}")
    print(f"  右手(1): 训练集 {(train_label==1).sum()}, 测试集 {(test_label==1).sum()}")

    return train_data, train_label, test_data, test_label


class TrainTestManager:
    def __init__(self, model_name, data_dir, dataset_type, n_sub,
                 batch_size=72, epochs=1000, lr=0.001,
                 heads=2, depth=6, emb_size=16, number_channel=22,
                 validate_ratio=0.3, N_AUG=3):

        self.model_name = model_name
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.n_sub = n_sub
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.heads = heads
        self.depth = depth
        self.emb_size = emb_size
        self.number_channel = number_channel
        self.number_class = 2  # 🔥 固定为2分类
        self.validate_ratio = validate_ratio
        self.N_AUG = N_AUG

        # 创建结果目录
        self.result_dir = f"{self.model_name}/"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def get_source_data(self):
        """加载2分类数据"""
        train_data, train_label, test_data, test_label = load_data_2class(
            self.data_dir, self.dataset_type, self.n_sub
        )
        return train_data, train_label, test_data, test_label

    def standard_normalize(self, dataset_train_data, dataset_test_data):
        """标准化"""
        scalar = dataset_train_data.std()
        mean_train_data = dataset_train_data.mean()
        dataset_train_data = (dataset_train_data - mean_train_data) / scalar
        dataset_test_data = (dataset_test_data - mean_train_data) / scalar
        return dataset_train_data, dataset_test_data

    def data_aug(self, temp_train_data, temp_train_label):
        """
        S&R数据增强 - 适配2分类

        输入:
            temp_train_data: (N, 22, 1000) - 没有通道维度
            temp_train_label: (N, 1)
        输出:
            tmp_aug_data: (N_aug, 1, 22, 1000) - 添加了通道维度
            tmp_aug_label: (N_aug, 1)
        """
        n_segments = 8
        segment_length = int(1000 / n_segments)

        # 2分类: 每类有约72个样本
        n_class0 = (temp_train_label == 0).sum()
        n_class1 = (temp_train_label == 1).sum()

        number_records_by_augmentation = n_class0 * self.N_AUG + n_class1 * self.N_AUG

        tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
        tmp_aug_label = np.zeros((number_records_by_augmentation, 1))

        aug_idx = 0

        # 对每个类别进行增强
        for class_id in [0, 1]:
            class_data = temp_train_data[temp_train_label.flatten() == class_id]  # (N, 22, 1000)
            n_samples = class_data.shape[0]

            for aug_round in range(self.N_AUG):
                for i in range(n_samples):
                    new_sample = np.zeros((1, self.number_channel, 1000))

                    for seg in range(n_segments):
                        random_idx = random.randint(0, n_samples - 1)
                        start = seg * segment_length
                        end = (seg + 1) * segment_length
                        # class_data是(N, 22, 1000), 直接索引
                        new_sample[0, :, start:end] = class_data[random_idx, :, start:end]

                    tmp_aug_data[aug_idx] = new_sample
                    tmp_aug_label[aug_idx] = class_id
                    aug_idx += 1

        print(f"  数据增强完成: {tmp_aug_data.shape[0]} 样本")
        return tmp_aug_data, tmp_aug_label

    def train(self):
        """训练流程"""
        print(f"\n{'='*60}")
        print(f"Subject {self.n_sub} - 2-Class Training (Left Hand vs Right Hand)")
        print(f"{'='*60}")

        # 加载数据
        train_data, train_label, test_data, test_label = self.get_source_data()

        # 标准化
        train_data, test_data = self.standard_normalize(train_data, test_data)

        # 划分训练集和验证集 (在添加通道维度之前)
        n_train_samples = train_data.shape[0]
        n_validate = int(self.validate_ratio * n_train_samples)

        indices = np.random.permutation(n_train_samples)
        val_indices = indices[:n_validate]
        train_indices = indices[n_validate:]

        train_data_split = train_data[train_indices]
        train_label_split = train_label[train_indices]

        # 数据增强 (输入是(N, 22, 1000),输出是(N_aug, 1, 22, 1000))
        aug_data, aug_label = self.data_aug(train_data_split, train_label_split)

        # 添加通道维度到验证集和测试集
        val_data = np.expand_dims(train_data[val_indices], axis=1)
        val_label = train_label[val_indices]
        test_data = np.expand_dims(test_data, axis=1)

        print(f"  训练集: {aug_data.shape[0]} (增强后)")
        print(f"  验证集: {val_data.shape[0]}")
        print(f"  测试集: {test_data.shape[0]}")

        # 创建模型
        model = EEGTransformer(
            heads=self.heads,
            emb_size=self.emb_size,
            depth=self.depth,
            n_classes=self.number_class,
            number_channel=self.number_channel,
            f1=8,
            kernel_size=64,
            D=2,
            pooling_size1=8,
            pooling_size2=8,
            dropout_rate=0.5,
            flatten_eeg1=240
        ).cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # 训练循环
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.epochs):
            model.train()

            # 打乱训练数据
            shuffle_idx = np.random.permutation(aug_data.shape[0])
            aug_data_shuffled = aug_data[shuffle_idx]
            aug_label_shuffled = aug_label[shuffle_idx]

            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx in range(0, len(aug_data_shuffled), self.batch_size):
                batch_data = aug_data_shuffled[batch_idx:batch_idx+self.batch_size]
                batch_label = aug_label_shuffled[batch_idx:batch_idx+self.batch_size]

                if len(batch_data) == 0:
                    continue

                batch_data = Variable(torch.from_numpy(batch_data).float()).cuda()
                batch_label = Variable(torch.from_numpy(batch_label).long().squeeze()).cuda()

                optimizer.zero_grad()
                _, outputs = model(batch_data)
                loss = criterion(outputs, batch_label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_label.size(0)
                train_correct += (predicted == batch_label).sum().item()

            train_acc = train_correct / train_total

            # 验证
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_idx in range(0, len(val_data), self.batch_size):
                    batch_data = val_data[batch_idx:batch_idx+self.batch_size]
                    batch_label = val_label[batch_idx:batch_idx+self.batch_size]

                    if len(batch_data) == 0:
                        continue

                    batch_data = Variable(torch.from_numpy(batch_data).float()).cuda()
                    batch_label = Variable(torch.from_numpy(batch_label).long().squeeze()).cuda()

                    _, outputs = model(batch_data)
                    loss = criterion(outputs, batch_label)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_label.size(0)
                    val_correct += (predicted == batch_label).sum().item()

            val_acc = val_correct / val_total

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model, f'{self.result_dir}/model_{self.n_sub}.pth')

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.6f}")

        print(f"\n最佳模型保存于Epoch {best_epoch}")

        # 测试
        model = torch.load(f'{self.result_dir}/model_{self.n_sub}.pth', weights_only=False)
        model.eval()

        test_data_tensor = Variable(torch.from_numpy(test_data).float()).cuda()

        with torch.no_grad():
            _, outputs = model(test_data_tensor)
            _, predictions = torch.max(outputs, 1)

        predictions = predictions.cpu().numpy()
        test_label_flat = test_label.flatten()

        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

        accuracy = accuracy_score(test_label_flat, predictions)
        precision = precision_score(test_label_flat, predictions, average='weighted', zero_division=0)
        recall = recall_score(test_label_flat, predictions, average='weighted', zero_division=0)
        f1 = f1_score(test_label_flat, predictions, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(test_label_flat, predictions)

        print(f"\n{'='*60}")
        print(f"Subject {self.n_sub} - 测试结果")
        print(f"{'='*60}")
        print(f"准确率: {accuracy*100:.2f}%")
        print(f"精确率: {precision*100:.2f}%")
        print(f"召回率: {recall*100:.2f}%")
        print(f"F1分数: {f1*100:.2f}%")
        print(f"Kappa: {kappa:.4f}")
        print(f"最佳Epoch: {best_epoch}")
        print(f"{'='*60}\n")

        return {
            'subject': self.n_sub,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'kappa': kappa,
            'best_epoch': best_epoch
        }


def main():
    """主函数 - 训练所有9个受试者"""

    # 配置
    model_name = "A_2class_leftright_heads_2_depth_6"
    data_dir = "../mymat_raw/"
    dataset_type = "A"

    print(f"\n{'='*60}")
    print(f"CTNet 2-Class Training (Left Hand vs Right Hand)")
    print(f"Model: {model_name}")
    print(f"Dataset: BCI Competition IV-2a")
    print(f"Classes: 2 (Left Hand=0, Right Hand=1)")
    print(f"{'='*60}\n")

    # 训练所有受试者
    results = []

    for subject in range(1, 10):
        manager = TrainTestManager(
            model_name=model_name,
            data_dir=data_dir,
            dataset_type=dataset_type,
            n_sub=subject,
            batch_size=72,
            epochs=1000,
            lr=0.001,
            heads=2,
            depth=6,
            emb_size=16,
            number_channel=22,
            validate_ratio=0.3,
            N_AUG=3
        )

        result = manager.train()
        results.append(result)

    # 保存结果
    df_results = pd.DataFrame(results)

    # 添加平均值和标准差
    mean_row = {
        'subject': 'Mean',
        'accuracy': df_results['accuracy'].mean(),
        'precision': df_results['precision'].mean(),
        'recall': df_results['recall'].mean(),
        'f1': df_results['f1'].mean(),
        'kappa': df_results['kappa'].mean(),
        'best_epoch': '-'
    }

    std_row = {
        'subject': 'Std',
        'accuracy': df_results['accuracy'].std(),
        'precision': df_results['precision'].std(),
        'recall': df_results['recall'].std(),
        'f1': df_results['f1'].std(),
        'kappa': df_results['kappa'].std(),
        'best_epoch': '-'
    }

    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # 保存到Excel
    excel_path = f"{model_name}/result_metric.xlsx"
    df_results.to_excel(excel_path, index=False)

    print(f"\n{'='*60}")
    print(f"所有受试者训练完成!")
    print(f"{'='*60}")
    print(df_results)
    print(f"\n结果已保存到: {excel_path}")
    print(f"\n平均准确率: {mean_row['accuracy']*100:.2f}% (±{std_row['accuracy']*100:.2f}%)")
    print(f"平均Kappa: {mean_row['kappa']:.4f} (±{std_row['kappa']:.4f})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 设置随机种子
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"随机种子: {seed}")

    # 开始训练
    main()
