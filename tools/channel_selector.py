"""
Automatic Channel Selection for 8-Channel EEG System (ADS1299)

This script provides three methods to select the optimal 8 channels from 22-channel BCI IV-2a data:
1. Prior Knowledge Method: Based on motor imagery research (C3, C4, Cz, etc.)
2. Mutual Information Method: Data-driven approach (RECOMMENDED)
3. Recursive Feature Elimination: Model-based selection

Author: Patrick
Date: 2025-10-19
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data_subject_dependent, numberClassChannel
import warnings
warnings.filterwarnings("ignore")


# BCI Competition IV-2a 的 22 个电极名称 (10-20国际系统)
CHANNEL_NAMES_2A = [
    'Fz',   # 0
    'FC3',  # 1
    'FC1',  # 2
    'FCz',  # 3
    'FC2',  # 4
    'FC4',  # 5
    'C5',   # 6
    'C3',   # 7
    'C1',   # 8
    'Cz',   # 9
    'C2',   # 10
    'C4',   # 11
    'C6',   # 12
    'CP3',  # 13
    'CP1',  # 14
    'CPz',  # 15
    'CP2',  # 16
    'CP4',  # 17
    'P1',   # 18
    'Pz',   # 19
    'P2',   # 20
    'POz'   # 21
]


class ChannelSelector:
    """8通道自动选择器"""

    def __init__(self, data_dir='./mymat_raw/', dataset_type='A', n_channels=8):
        """
        初始化通道选择器

        参数:
            data_dir: 数据目录
            dataset_type: 'A' for BCI IV-2a, 'B' for BCI IV-2b
            n_channels: 要选择的通道数量 (默认8)
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.n_channels = n_channels
        self.channel_names = CHANNEL_NAMES_2A if dataset_type == 'A' else ['C3', 'Cz', 'C4']

        # 存储结果
        self.mi_scores = None
        self.selected_channels = {}

    def method1_prior_knowledge(self):
        """
        方法1: 基于先验知识选择通道

        根据运动想象研究,选择与运动皮层相关的关键电极

        返回:
            selected_indices: 选中的通道索引
            selected_names: 选中的通道名称
        """
        print("\n" + "="*60)
        print("方法1: 基于先验知识的通道选择")
        print("="*60)

        # 运动想象任务的关键电极 (基于神经科学研究)
        priority_channels = ['C3', 'C4', 'Cz', 'FCz', 'CP1', 'CP2', 'FC3', 'FC4']

        selected_indices = []
        selected_names = []

        for ch_name in priority_channels[:self.n_channels]:
            if ch_name in self.channel_names:
                idx = self.channel_names.index(ch_name)
                selected_indices.append(idx)
                selected_names.append(ch_name)

        print(f"\n✅ 选中的{len(selected_indices)}个通道:")
        for i, (idx, name) in enumerate(zip(selected_indices, selected_names)):
            print(f"   {i+1}. {name:6s} (索引: {idx:2d})")

        self.selected_channels['prior_knowledge'] = {
            'indices': selected_indices,
            'names': selected_names
        }

        return selected_indices, selected_names

    def method2_mutual_information(self, subject=1, use_all_subjects=True):
        """
        方法2: 基于互信息的通道选择 (推荐)

        计算每个通道与运动想象标签的互信息,选择信息量最大的通道

        参数:
            subject: 受试者编号 (1-9)
            use_all_subjects: 是否使用所有受试者的数据 (推荐)

        返回:
            selected_indices: 选中的通道索引
            selected_names: 选中的通道名称
        """
        print("\n" + "="*60)
        print("方法2: 基于互信息的通道选择 (数据驱动)")
        print("="*60)

        if self.dataset_type != 'A':
            print("⚠️  此方法仅支持2a数据集 (22通道)")
            return None, None

        # 加载数据
        all_train_data = []
        all_train_label = []

        subjects = range(1, 10) if use_all_subjects else [subject]

        print(f"\n📊 正在加载数据...")
        for sub in subjects:
            train_data, train_label, _, _ = load_data_subject_dependent(
                self.data_dir, self.dataset_type, sub
            )
            all_train_data.append(train_data)
            # train_label 的形状是 (n_trials, 1), 需要展平为 (n_trials,)
            all_train_label.append(train_label.flatten())
            print(f"   受试者 {sub}: {train_data.shape[0]} trials")

        # 合并数据
        X = np.vstack(all_train_data)  # (N_trials, 22, 1000)
        y = np.hstack(all_train_label)  # (N_trials,)

        print(f"\n📈 总数据: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints")

        # 计算每个通道的互信息
        print(f"\n⚙️  计算互信息...")
        mi_scores = np.zeros(X.shape[1])  # 22个通道

        for ch in range(X.shape[1]):
            # 对每个通道提取特征
            ch_data = X[:, ch, :]  # (n_trials, n_timepoints)
            ch_features = self._extract_single_channel_features(ch_data)

            # 计算该通道的互信息
            mi = mutual_info_classif(ch_features, y - 1, random_state=42)
            # 取平均作为该通道的综合互信息得分
            mi_scores[ch] = np.mean(mi)

        self.mi_scores = mi_scores

        # 选择Top-K通道
        top_indices = np.argsort(mi_scores)[-self.n_channels:][::-1]
        selected_names = [self.channel_names[i] for i in top_indices]

        # 显示结果
        print(f"\n✅ 互信息排名 (Top {self.n_channels}):")
        print(f"{'排名':<6} {'通道':<8} {'索引':<6} {'互信息':<12} {'归一化得分':<12}")
        print("-" * 60)

        max_mi = mi_scores.max()
        for rank, idx in enumerate(top_indices):
            mi = mi_scores[idx]
            normalized = mi / max_mi * 100
            print(f"{rank+1:<6} {self.channel_names[idx]:<8} {idx:<6} {mi:<12.6f} {normalized:<12.2f}%")

        # 保存结果
        self.selected_channels['mutual_information'] = {
            'indices': top_indices.tolist(),
            'names': selected_names,
            'scores': mi_scores[top_indices].tolist()
        }

        return top_indices.tolist(), selected_names

    def method3_rfe(self, subject=1, use_all_subjects=False):
        """
        方法3: 递归特征消除 (基于模型)

        使用随机森林模型逐步消除不重要的通道
        注意: 此方法计算量较大

        参数:
            subject: 受试者编号 (1-9)
            use_all_subjects: 是否使用所有受试者

        返回:
            selected_indices: 选中的通道索引
            selected_names: 选中的通道名称
        """
        print("\n" + "="*60)
        print("方法3: 递归特征消除 (RFE)")
        print("="*60)

        if self.dataset_type != 'A':
            print("⚠️  此方法仅支持2a数据集")
            return None, None

        # 加载数据
        all_train_data = []
        all_train_label = []

        subjects = range(1, 10) if use_all_subjects else [subject]

        print(f"\n📊 正在加载数据...")
        for sub in subjects:
            train_data, train_label, _, _ = load_data_subject_dependent(
                self.data_dir, self.dataset_type, sub
            )
            all_train_data.append(train_data)
            # train_label 的形状是 (n_trials, 1), 需要展平为 (n_trials,)
            all_train_label.append(train_label.flatten())

        X = np.vstack(all_train_data)
        y = np.hstack(all_train_label)

        # 提取特征
        print(f"\n⚙️  提取通道特征...")
        channel_features = self._extract_channel_features(X)

        # RFE (在特征级别选择,每个通道有5个特征)
        print(f"⚙️  执行递归特征消除 (可能需要几分钟)...")
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        # 选择 n_channels * 5 个特征 (每个通道5个特征)
        selector = RFE(estimator, n_features_to_select=self.n_channels * 5, step=5)
        selector.fit(channel_features, y - 1)

        # 获取选中的特征索引,并映射回通道索引
        selected_feature_mask = selector.support_
        selected_feature_indices = np.where(selected_feature_mask)[0]

        # 每5个特征对应一个通道,计算通道索引
        selected_channel_indices = np.unique(selected_feature_indices // 5)

        # 如果选中的通道数超过要求,按照ranking选择top-k
        if len(selected_channel_indices) > self.n_channels:
            # 计算每个通道的平均ranking
            channel_rankings = np.zeros(X.shape[1])
            for ch in range(X.shape[1]):
                feature_start = ch * 5
                feature_end = feature_start + 5
                channel_rankings[ch] = np.mean(selector.ranking_[feature_start:feature_end])

            # 选择ranking最小的通道
            selected_channel_indices = np.argsort(channel_rankings)[:self.n_channels]

        selected_indices = selected_channel_indices.tolist()
        selected_names = [self.channel_names[i] for i in selected_indices]

        # 显示结果
        print(f"\n✅ RFE 选中的 {len(selected_indices)} 个通道:")
        for i, (idx, name) in enumerate(zip(selected_indices, selected_names)):
            ranking = selector.ranking_[idx]
            print(f"   {i+1}. {name:6s} (索引: {idx:2d}, 排名: {ranking})")

        self.selected_channels['rfe'] = {
            'indices': selected_indices,
            'names': selected_names
        }

        return selected_indices, selected_names

    def _extract_single_channel_features(self, ch_data):
        """
        从单个通道的EEG数据中提取特征

        参数:
            ch_data: (n_trials, n_timepoints)

        返回:
            features: (n_trials, n_features)
        """
        # 时域特征
        mean = np.mean(ch_data, axis=1, keepdims=True)
        std = np.std(ch_data, axis=1, keepdims=True)
        variance = np.var(ch_data, axis=1, keepdims=True)
        max_val = np.max(ch_data, axis=1, keepdims=True)
        min_val = np.min(ch_data, axis=1, keepdims=True)

        # 合并特征 (n_trials, 5)
        features = np.hstack([mean, std, variance, max_val, min_val])

        return features

    def _extract_channel_features(self, X):
        """
        从EEG数据中提取所有通道的特征 (用于RFE方法)

        参数:
            X: (n_trials, n_channels, n_timepoints)

        返回:
            features: (n_trials, n_channels * n_features)
        """
        n_trials, n_channels, n_timepoints = X.shape
        features = []

        for ch in range(n_channels):
            ch_data = X[:, ch, :]  # (n_trials, n_timepoints)
            ch_features = self._extract_single_channel_features(ch_data)
            features.append(ch_features)

        # (n_trials, n_channels * 5)
        features = np.hstack(features)

        return features

    def visualize_mi_scores(self, save_path=None):
        """
        可视化互信息得分

        参数:
            save_path: 保存路径 (可选)
        """
        if self.mi_scores is None:
            print("⚠️  请先运行 method2_mutual_information()")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # 条形图
        indices = np.arange(len(self.mi_scores))
        colors = ['red' if i in self.selected_channels['mutual_information']['indices']
                  else 'lightgray' for i in indices]

        ax1.bar(indices, self.mi_scores, color=colors, alpha=0.7)
        ax1.set_xlabel('Channel Index', fontsize=12)
        ax1.set_ylabel('Mutual Information Score', fontsize=12)
        ax1.set_title('Mutual Information Scores for Each Channel', fontsize=14, fontweight='bold')
        ax1.set_xticks(indices)
        ax1.set_xticklabels(self.channel_names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # 标注选中的通道
        selected_idx = self.selected_channels['mutual_information']['indices']
        for idx in selected_idx:
            ax1.text(idx, self.mi_scores[idx], '★', ha='center', va='bottom',
                    fontsize=16, color='red', fontweight='bold')

        # 热力图 (按重要性排序)
        sorted_indices = np.argsort(self.mi_scores)[::-1]
        sorted_scores = self.mi_scores[sorted_indices]
        sorted_names = [self.channel_names[i] for i in sorted_indices]

        # 创建数据用于热力图
        heatmap_data = sorted_scores.reshape(-1, 1)

        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd',
                   yticklabels=sorted_names, xticklabels=['MI Score'],
                   cbar_kws={'label': 'Mutual Information'}, ax=ax2)
        ax2.set_title('Channel Importance Ranking', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 图表已保存到: {save_path}")

        plt.show()

    def save_results(self, output_file='selected_channels.txt'):
        """
        保存通道选择结果到文件

        参数:
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("8-Channel Selection Results for ADS1299\n")
            f.write("="*60 + "\n\n")

            for method_name, result in self.selected_channels.items():
                f.write(f"\n方法: {method_name}\n")
                f.write("-"*60 + "\n")
                f.write(f"选中的通道索引: {result['indices']}\n")
                f.write(f"选中的通道名称: {result['names']}\n")

                if 'scores' in result:
                    f.write(f"互信息得分: {result['scores']}\n")

                f.write("\n")

        print(f"\n💾 结果已保存到: {output_file}")

    def compare_methods(self):
        """对比不同方法的结果"""
        if len(self.selected_channels) < 2:
            print("⚠️  请先运行至少两种选择方法")
            return

        print("\n" + "="*60)
        print("方法对比")
        print("="*60)

        methods = list(self.selected_channels.keys())

        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                set1 = set(self.selected_channels[method1]['indices'])
                set2 = set(self.selected_channels[method2]['indices'])

                overlap = set1 & set2
                overlap_pct = len(overlap) / self.n_channels * 100

                print(f"\n{method1} vs {method2}:")
                print(f"  重叠通道: {len(overlap)}/{self.n_channels} ({overlap_pct:.1f}%)")
                if overlap:
                    overlap_names = [self.channel_names[i] for i in sorted(overlap)]
                    print(f"  共同通道: {overlap_names}")


def main():
    """主函数 - 演示三种方法"""

    print("\n" + "="*60)
    print("自动8通道选择器 - 适配ADS1299")
    print("="*60)

    # 初始化选择器
    selector = ChannelSelector(
        data_dir='./mymat_raw/',
        dataset_type='A',
        n_channels=8
    )

    # 方法1: 先验知识
    print("\n开始执行方法1...")
    indices1, names1 = selector.method1_prior_knowledge()

    # 方法2: 互信息 (推荐)
    print("\n开始执行方法2 (推荐)...")
    indices2, names2 = selector.method2_mutual_information(use_all_subjects=True)

    # 方法3: RFE (可选,计算量大)
    user_input = input("\n是否执行方法3 (RFE,计算量较大,可能需要5-10分钟)? [y/N]: ")
    if user_input.lower() == 'y':
        print("\n开始执行方法3...")
        indices3, names3 = selector.method3_rfe(use_all_subjects=True)

    # 对比方法
    selector.compare_methods()

    # 可视化
    print("\n生成可视化图表...")
    selector.visualize_mi_scores(save_path='channel_selection_results.png')

    # 保存结果
    selector.save_results('selected_channels.txt')

    print("\n" + "="*60)
    print("✅ 通道选择完成!")
    print("="*60)
    print("\n📌 推荐使用方法2 (互信息) 的结果:")
    print(f"   索引: {indices2}")
    print(f"   名称: {names2}")
    print("\n💡 请将这些索引用于 main_8_channels.py")


if __name__ == "__main__":
    main()