"""
Quick validation training script for subjects 1, 3, 7 only.
Tests the channel-attention feature branch without running all 9 subjects.
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

from torchsummary import summary
import torch
from torch.backends import cudnn
import warnings
warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True

from model import EEGTransformer
from utils import calMetrics, numberClassChannel
from train import ExP

# ---- Hyperparameters (same as train.py __main__) ----
DATA_DIR = r'./mymat_raw/'
EVALUATE_MODE = 'LOSO-No'

SUBJECTS = [1, 3, 7]  # Only these 3 subjects for quick validation
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
L1_LAMBDA = 1e-4

if EVALUATE_MODE != 'LOSO':
    EEGNet1_DROPOUT_RATE = 0.5
else:
    EEGNet1_DROPOUT_RATE = 0.25

number_class, number_channel = numberClassChannel(TYPE)
RESULT_NAME = "{}_heads_{}_depth_{}_quick".format(TYPE, HEADS, DEPTH)

# Print model summary once
sModel = EEGTransformer(
    heads=HEADS,
    emb_size=EMB_DIM,
    depth=DEPTH,
    database_type=TYPE,
    eeg1_f1=EEGNet1_F1,
    eeg1_D=EEGNet1_D,
    eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
    eeg1_pooling_size1=EEGNet1_POOL_SIZE1,
    eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
    eeg1_dropout_rate=EEGNet1_DROPOUT_RATE,
    eeg1_number_channel=number_channel,
    flatten_eeg1=FLATTEN_EEGNet1,
).cuda()
summary(sModel, (1, number_channel, 1000))
del sModel
torch.cuda.empty_cache()

if not os.path.exists(RESULT_NAME):
    os.makedirs(RESULT_NAME)

print("=" * 60)
print("Quick validation: subjects", SUBJECTS)
print("Start time:", time.asctime(time.localtime(time.time())))
print("=" * 60)

subjects_result = []
best_epochs = []

for sub_idx, sub in enumerate(SUBJECTS):
    starttime = datetime.datetime.now()
    seed_n = np.random.randint(2024)
    print('\nseed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    print('Subject %d' % sub)
    exp = ExP(sub, DATA_DIR, RESULT_NAME, EPOCHS, N_AUG, N_SEG, gpus,
              evaluate_mode=EVALUATE_MODE,
              heads=HEADS,
              emb_size=EMB_DIM,
              depth=DEPTH,
              dataset_type=TYPE,
              eeg1_f1=EEGNet1_F1,
              eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
              eeg1_D=EEGNet1_D,
              eeg1_pooling_size1=EEGNet1_POOL_SIZE1,
              eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
              eeg1_dropout_rate=EEGNet1_DROPOUT_RATE,
              flatten_eeg1=FLATTEN_EEGNet1,
              validate_ratio=validate_ratio,
              l1_lambda=L1_LAMBDA,
              )

    testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
    true_cpu = Y_true.cpu().numpy().astype(int)
    pred_cpu = Y_pred.cpu().numpy().astype(int)

    accuracy, precision, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
    subject_result = {
        'subject': sub,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'kappa': kappa * 100,
        'best_epoch': best_epoch,
    }
    subjects_result.append(subject_result)
    best_epochs.append(best_epoch)

    print(' THE BEST ACCURACY IS ' + str(testAcc) + "\tkappa is " + str(kappa))
    endtime = datetime.datetime.now()
    print('Subject %d duration: ' % sub + str(endtime - starttime))

    # Clean up GPU memory between subjects
    del exp
    torch.cuda.empty_cache()

# Print summary
print("\n" + "=" * 60)
print("QUICK VALIDATION RESULTS SUMMARY")
print("=" * 60)
df_result = pd.DataFrame(subjects_result)
print(df_result.to_string(index=False))
print("-" * 60)
print("Mean accuracy: {:.2f}%".format(df_result['accuracy'].mean()))
print("Std accuracy:  {:.2f}%".format(df_result['accuracy'].std()))
print("Mean kappa:    {:.2f}%".format(df_result['kappa'].mean()))
print("Best epochs:   ", best_epochs)
print("=" * 60)

# Check for channel attention .npy files
print("\nChannel attention weight files:")
for sub in SUBJECTS:
    ca_path = RESULT_NAME + '/channel_attention_sub{}.npy'.format(sub)
    if os.path.exists(ca_path):
        ca_weights = np.load(ca_path)
        print("  Sub {}: {} (shape: {})".format(sub, ca_weights, ca_weights.shape))
    else:
        print("  Sub {}: NOT FOUND".format(sub))

print("\nEnd time:", time.asctime(time.localtime(time.time())))
