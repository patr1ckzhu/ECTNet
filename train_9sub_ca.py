"""Channel Attention: all 9 subjects, fixed seed=42, l1_lambda=1e-4."""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import random
import time
import torch
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
from train import ExP
from utils import calMetrics

FIXED_SEED = 42
result_dir = 'compare_9sub_ca'
os.makedirs(result_dir, exist_ok=True)

results = []
for sub in range(1, 10):
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)

    print(f"\n{'='*60}")
    print(f"  CHANNEL ATTENTION - Subject {sub}, Seed {FIXED_SEED}")
    print(f"{'='*60}")

    exp = ExP(sub, r'./mymat_raw/', result_dir, 1000, 3, 8, [0],
              evaluate_mode='LOSO-No', heads=2, emb_size=16, depth=6,
              dataset_type='A', eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
              eeg1_pooling_size1=8, eeg1_pooling_size2=8,
              eeg1_dropout_rate=0.5, flatten_eeg1=240, validate_ratio=0.3,
              l1_lambda=1e-4)

    start = time.time()
    testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
    elapsed = time.time() - start
    acc, prec, rec, f1, kappa = calMetrics(Y_true.cpu().numpy().astype(int), Y_pred.cpu().numpy().astype(int))
    results.append({'sub': sub, 'acc': acc*100, 'kappa': kappa*100, 'epoch': best_epoch, 'time': elapsed})
    print(f"\nCHANNEL ATTENTION S{sub}: acc={acc*100:.2f}%, kappa={kappa*100:.2f}%, epoch={best_epoch}, time={elapsed:.1f}s")

    # Print channel attention weights
    ca_path = result_dir + f'/channel_attention_sub{sub}.npy'
    if os.path.exists(ca_path):
        ca_w = np.load(ca_path)
        print(f"  Channel weights: {ca_w}")
    else:
        print(f"  Channel weights file not found at {ca_path}")

    del exp
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"  CHANNEL ATTENTION SUMMARY (All 9 Subjects, Seed {FIXED_SEED})")
print(f"{'='*60}")
accs = [r['acc'] for r in results]
kappas = [r['kappa'] for r in results]
for r in results:
    print(f"  S{r['sub']}: acc={r['acc']:.2f}%, kappa={r['kappa']:.2f}%, epoch={r['epoch']}, time={r['time']:.1f}s")
print(f"  {'---'*17}")
print(f"  Mean acc:   {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
print(f"  Mean kappa: {np.mean(kappas):.2f}% +/- {np.std(kappas):.2f}%")
print(f"{'='*60}")
