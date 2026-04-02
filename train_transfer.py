"""
Transfer Learning for ECTNet: Pretrain on BCI IV-2b, fine-tune on custom data.

Usage:
    # Phase 1: Pretrain on all 9 B-subjects (~3600 trials)
    python train_transfer.py pretrain

    # Phase 2: Fine-tune on custom C data with frozen CNN
    python train_transfer.py finetune --pretrained pretrained_B/model_pretrained.pth --freeze cnn

    # Baseline: Train C from scratch (for comparison)
    python train_transfer.py baseline
"""

import argparse
import os
import time
import platform
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import EEGTransformer
from utils import load_data, numberClassChannel, calMetrics

# ─── Constants ───────────────────────────────────────────────────────────────

NUMBER_CLASS = 2
NUMBER_CHANNEL = 3
SEED = 42

# Model config (matching train.py for B/C)
MODEL_CONFIG = dict(
    heads=2, emb_size=16, depth=6,
    database_type='B',
    eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
    eeg1_pooling_size1=8, eeg1_pooling_size2=8,
    eeg1_dropout_rate=0.5,
    eeg1_number_channel=NUMBER_CHANNEL,
    flatten_eeg1=240,
)


# ─── Euclidean Alignment ─────────────────────────────────────────────────────

def compute_ea_matrix(X):
    """Compute Euclidean Alignment whitening matrix R^(-1/2).

    Args:
        X: (n_trials, n_channels, n_timepoints)
    Returns:
        R_inv_sqrt: (n_channels, n_channels)
    """
    n_trials, n_ch, n_t = X.shape
    R = np.zeros((n_ch, n_ch))
    for i in range(n_trials):
        R += X[i] @ X[i].T
    R /= n_trials
    R += 1e-8 * np.eye(n_ch)  # regularize

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return R_inv_sqrt.astype(np.float32)


def apply_ea(X, R_inv_sqrt):
    """Apply Euclidean Alignment: X_aligned = R^(-1/2) @ X.

    Args:
        X: (n_trials, n_channels, n_timepoints)
        R_inv_sqrt: (n_channels, n_channels)
    Returns:
        X_aligned: same shape as X
    """
    return np.einsum('ij,njt->nit', R_inv_sqrt, X)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_all_B_data(data_dir='./mymat_raw/', apply_filter=False):
    """Load ALL 9 B subjects' T+E data for subject-independent pretraining.

    Returns:
        data: (N, 3, 1000) in Volts
        labels: (N,) values in {1, 2}
    """
    all_data, all_labels = [], []
    for sub in range(1, 10):
        for mode in ['train', 'test']:
            data, label = load_data(data_dir, 'B', sub, mode=mode, apply_filter=apply_filter)
            all_data.append(data)
            all_labels.append(label.flatten())
    return np.concatenate(all_data), np.concatenate(all_labels)


def load_all_B_data_ea(data_dir='./mymat_raw/', apply_filter=False):
    """Load all 9 B subjects with per-subject Euclidean Alignment."""
    all_data, all_labels = [], []
    for sub in range(1, 10):
        sub_data, sub_labels = [], []
        for mode in ['train', 'test']:
            data, label = load_data(data_dir, 'B', sub, mode=mode, apply_filter=apply_filter)
            sub_data.append(data)
            sub_labels.append(label.flatten())
        sub_data = np.concatenate(sub_data)
        sub_labels = np.concatenate(sub_labels)

        R_inv_sqrt = compute_ea_matrix(sub_data)
        sub_data = apply_ea(sub_data, R_inv_sqrt)
        print(f"  B-S{sub}: {sub_data.shape[0]} trials, EA applied")

        all_data.append(sub_data)
        all_labels.append(sub_labels)
    return np.concatenate(all_data), np.concatenate(all_labels)


def load_C_data(data_dir='./mymat_custom/', subject=1, apply_filter=False):
    """Load custom C data, converting µV → V to match B dataset units.

    Returns:
        train_data, train_labels, test_data, test_labels
        Data in Volts, labels in {1, 2}
    """
    train_data, train_label = load_data(data_dir, 'C', subject, mode='train', apply_filter=apply_filter)
    test_data, test_label = load_data(data_dir, 'C', subject, mode='test', apply_filter=apply_filter)

    # µV → V (B dataset is in V from MNE epochs.get_data())
    train_data = train_data * 1e-6
    test_data = test_data * 1e-6

    return train_data, train_label.flatten(), test_data, test_label.flatten()


# ─── Data Augmentation ───────────────────────────────────────────────────────

def interaug(data, labels, batch_size, number_aug=3, number_seg=8):
    """Segmentation and Reconstruction (S&R) data augmentation.

    Args:
        data: (N, 1, C, 1000) normalized training data
        labels: (N,) labels in {1, 2, ...}
        batch_size: base batch size
        number_aug: augmentation multiplier
        number_seg: number of segments for S&R

    Returns:
        (aug_data_tensor, aug_label_tensor) on CUDA, labels 0-indexed
    """
    n_per_class = number_aug * (batch_size // NUMBER_CLASS)
    seg_len = 1000 // number_seg
    aug_data, aug_label = [], []

    for cls in range(NUMBER_CLASS):
        cls_data = data[labels == cls + 1]  # (n_cls, 1, ch, 1000)
        n_cls = cls_data.shape[0]
        if n_cls == 0:
            continue

        rand_idx = np.random.randint(0, n_cls, (n_per_class, number_seg))
        tmp = np.empty((n_per_class, 1, NUMBER_CHANNEL, 1000), dtype=cls_data.dtype)
        for s in range(number_seg):
            start = s * seg_len
            end = start + seg_len
            tmp[:, :, :, start:end] = cls_data[rand_idx[:, s], :, :, start:end]

        aug_data.append(tmp)
        aug_label.append(np.full(n_per_class, cls + 1))

    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    shuffle = np.random.permutation(len(aug_data))

    return (torch.from_numpy(aug_data[shuffle]).float().cuda(),
            torch.from_numpy(aug_label[shuffle] - 1).long().cuda())


def online_augment(batch, noise_std=0.1, scale_range=(0.8, 1.2), shift_max=50):
    """Apply random online augmentations to a batch of EEG data.

    Args:
        batch: Tensor (B, 1, C, T) on CUDA, already normalized
        noise_std: Gaussian noise std relative to signal std
        scale_range: (min, max) for random amplitude scaling
        shift_max: max samples for random time shift

    Returns:
        augmented batch (same shape, same device)
    """
    B, _, C, T = batch.shape

    # Gaussian noise (per-trial random std)
    if noise_std > 0:
        noise_level = torch.rand(B, 1, 1, 1, device=batch.device) * noise_std
        batch = batch + torch.randn_like(batch) * noise_level

    # Random amplitude scaling (per-trial)
    lo, hi = scale_range
    scale = lo + torch.rand(B, 1, 1, 1, device=batch.device) * (hi - lo)
    batch = batch * scale

    # Random time shift (circular, per-trial)
    if shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max + 1, (B,))
        for i in range(B):
            if shifts[i] != 0:
                batch[i] = torch.roll(batch[i], shifts[i].item(), dims=-1)

    return batch


# ─── Layer Freezing ──────────────────────────────────────────────────────────

def apply_freeze(model, strategy):
    """Freeze model parameters according to strategy."""
    if strategy == 'none':
        return

    # Freeze CNN
    if strategy in ('cnn', 'cnn+transformer4'):
        for param in model.cnn.parameters():
            param.requires_grad = False

    # Freeze transformer blocks 0-3
    if strategy == 'cnn+transformer4':
        for i in range(4):
            for param in model.trans[i].parameters():
                param.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Freeze strategy: {strategy} — {frozen}/{total} param groups frozen, {trainable} trainable params")


def set_train_mode(model, strategy):
    """Set model to train mode. BN layers stay in train mode (AdaBN) so their
    running statistics adapt to the new domain, even when Conv weights are frozen."""
    model.train()


def build_optimizer(model, strategy, lr, diff_lr=False):
    """Build optimizer. Supports differential LR for transfer learning."""
    if diff_lr:
        param_groups = [
            {'params': model.cnn.parameters(), 'lr': lr * 0.1},
            {'params': model.position.parameters(), 'lr': lr * 0.3},
            {'params': model.trans.parameters(), 'lr': lr * 0.3},
            {'params': model.classification.parameters(), 'lr': lr},
        ]
        # Filter out frozen params
        param_groups = [
            {**g, 'params': [p for p in g['params'] if p.requires_grad]}
            for g in param_groups
        ]
        param_groups = [g for g in param_groups if g['params']]
        return torch.optim.Adam(param_groups, betas=(0.5, 0.999))
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))


# ─── Training Core ───────────────────────────────────────────────────────────

def train_loop(model, train_data, train_labels, test_data, test_labels,
               epochs, lr, batch_size, val_ratio, number_aug, number_seg,
               save_path, freeze_strategy='none', l1_lambda=0,
               pretrained_norm=None, ea_matrix=None,
               label_smoothing=0.0, cosine_lr=False, diff_lr=False,
               online_aug=False):
    """Unified training loop for pretrain, finetune, and baseline.

    Args:
        train_data: (N, 3, 1000) raw data (already in Volts)
        train_labels: (N,) labels in {1, 2}
        test_data: (N_test, 3, 1000) or None
        test_labels: (N_test,) or None
        pretrained_norm: (mean, std) tuple to use pretrained normalization stats
        label_smoothing: label smoothing factor (0.0 = off)
        cosine_lr: use cosine annealing LR schedule
        diff_lr: use differential learning rates (CNN < Transformer < Classifier)
        online_aug: apply online augmentation (noise, scaling, time shift)

    Returns:
        dict with accuracy, kappa, norm_mean, norm_std, best_epoch
    """
    # Expand dims: (N, 3, 1000) → (N, 1, 3, 1000)
    train_data = np.expand_dims(train_data, axis=1)
    if test_data is not None:
        test_data = np.expand_dims(test_data, axis=1)

    # Shuffle
    perm = np.random.permutation(len(train_data))
    train_data = train_data[perm]
    train_labels = train_labels[perm]

    # Global standardization
    if pretrained_norm is not None:
        norm_mean, norm_std = pretrained_norm
        print(f"Using PRETRAINED normalization stats")
    else:
        norm_mean = np.mean(train_data)
        norm_std = np.std(train_data)
    train_data = (train_data - norm_mean) / norm_std
    if test_data is not None:
        test_data = (test_data - norm_mean) / norm_std

    print(f"Normalization: mean={norm_mean:.6e}, std={norm_std:.6e}")
    print(f"Train: {train_data.shape}, labels: {np.unique(train_labels, return_counts=True)}")

    # Train/val split
    n_total = len(train_data)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val
    split_perm = torch.randperm(n_total)
    train_idx = split_perm[:n_train]
    val_idx = split_perm[n_train:]

    all_data_np = train_data  # interaug samples from full dataset (incl. val)
    all_labels_np = train_labels

    img = torch.from_numpy(train_data)
    label = torch.from_numpy(train_labels - 1)  # 0-indexed for CrossEntropyLoss
    dataset = torch.utils.data.TensorDataset(img[train_idx], label[train_idx])
    val_img = img[val_idx].float().cuda()
    val_label = label[val_idx].long().cuda()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Apply freezing
    apply_freeze(model, freeze_strategy)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, freeze_strategy, lr, diff_lr=diff_lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine_lr else None

    # torch.compile on Linux
    if platform.system() != 'Windows':
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    # Training
    best_epoch = 0
    min_loss = float('inf')
    val_interval = 1 if epochs <= 50 else 5
    train_start = time.time()

    for e in range(epochs):
        set_train_mode(model, freeze_strategy)

        for batch_img, batch_label in dataloader:
            batch_img = batch_img.float().cuda()
            batch_label = batch_label.long().cuda()

            # S&R augmentation
            aug_data, aug_label = interaug(all_data_np, all_labels_np, batch_size,
                                           number_aug=number_aug, number_seg=number_seg)
            batch_img = torch.cat((batch_img, aug_data))
            batch_label = torch.cat((batch_label, aug_label))

            # Online augmentation (noise, scaling, time shift)
            if online_aug:
                batch_img = online_augment(batch_img)

            _, outputs = compiled_model(batch_img)
            loss = criterion(outputs, batch_label)

            # L1 regularization on channel attention
            if l1_lambda > 0:
                l1_loss = sum(p.abs().sum() for p in model.get_channel_attention_params())
                loss = loss + l1_lambda * l1_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Validation
        if (e + 1) % val_interval == 0 or e == epochs - 1:
            compiled_model.eval()
            with torch.no_grad():
                _, val_outputs = compiled_model(val_img)
                val_loss = criterion(val_outputs, val_label)
                val_acc = (val_outputs.argmax(dim=1) == val_label).float().mean().item()

            if val_loss < min_loss:
                min_loss = val_loss
                best_epoch = e
                torch.save({
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'norm_mean': norm_mean,
                    'norm_std': norm_std,
                    'ea_matrix': ea_matrix,
                }, save_path)

            if (e + 1) % 50 == 0 or e == epochs - 1:
                print(f"  Epoch {e:4d}  val_acc={val_acc:.4f}  val_loss={val_loss.item():.6f}"
                      f"  train_loss={loss.item():.6f}  {'*BEST' if e == best_epoch else ''}")

    total_time = time.time() - train_start
    print(f"Training done: {total_time:.1f}s, best_epoch={best_epoch}")

    # Evaluate on test set
    result = {'norm_mean': norm_mean, 'norm_std': norm_std, 'best_epoch': best_epoch}

    if test_data is not None:
        checkpoint = torch.load(save_path, weights_only=False)
        model = checkpoint['model'].cuda()
        model.eval()

        test_t = torch.from_numpy(test_data).float().cuda()
        test_l = torch.from_numpy(test_labels - 1).long()

        with torch.no_grad():
            outputs_list = []
            for i in range(0, len(test_t), batch_size):
                _, out = model(test_t[i:i+batch_size])
                outputs_list.append(out)
            outputs = torch.cat(outputs_list)

        y_pred = outputs.argmax(dim=1).cpu().numpy()
        y_true = test_l.numpy()
        acc, prec, rec, f1, kappa = calMetrics(y_true, y_pred)
        print(f"  TEST — acc={acc*100:.2f}%  kappa={kappa*100:.2f}%  prec={prec*100:.2f}%  "
              f"rec={rec*100:.2f}%  f1={f1*100:.2f}%")
        result.update(accuracy=acc, kappa=kappa, precision=prec, recall=rec, f1=f1)

        # Channel attention weights
        with torch.no_grad():
            ca_list = []
            for i in range(0, len(test_t), batch_size):
                model(test_t[i:i+batch_size])
                w = model.get_channel_attention_weights()
                if w is not None:
                    ca_list.append(w.cpu())
            if ca_list:
                ca = torch.cat(ca_list).mean(dim=0).numpy()
                print(f"  Channel attention: {ca}")

    return result


# ─── Mode: Pretrain ──────────────────────────────────────────────────────────

def pretrain(args):
    """Pretrain on all 9 B subjects pooled together."""
    print("=" * 60)
    print("  PRETRAIN: Subject-independent pretraining on BCI IV-2b")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    filename = 'model_pretrained_ea.pth' if args.ea else 'model_pretrained.pth'
    save_path = os.path.join(args.output_dir, filename)

    if args.ea:
        print("  EA: per-subject Euclidean Alignment")
        data, labels = load_all_B_data_ea()
    else:
        data, labels = load_all_B_data()
    print(f"Loaded B data: {data.shape}, labels: {np.unique(labels, return_counts=True)}")

    # Split off 10% as held-out test for sanity check
    n = len(data)
    perm = np.random.permutation(n)
    n_test = int(0.1 * n)
    test_data, test_labels = data[perm[:n_test]], labels[perm[:n_test]]
    train_data, train_labels = data[perm[n_test:]], labels[perm[n_test:]]

    model = EEGTransformer(**MODEL_CONFIG).cuda()
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    result = train_loop(
        model, train_data, train_labels, test_data, test_labels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_ratio=args.val_ratio, number_aug=3, number_seg=8,
        save_path=save_path, freeze_strategy='none', l1_lambda=1e-4,
    )

    print(f"\nPretrained model saved to: {save_path}")
    if 'accuracy' in result:
        print(f"Held-out test acc: {result['accuracy']*100:.2f}%")
    return result


# ─── Mode: Finetune ──────────────────────────────────────────────────────────

def finetune(args):
    """Fine-tune pretrained model on custom C data."""
    print("=" * 60)
    print(f"  FINETUNE: Transfer learning (freeze={args.freeze})")
    print("=" * 60)

    ea_suffix = '_ea' if args.ea else ''
    output_dir = f"{args.output_dir}_freeze_{args.freeze}{ea_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    seed_suffix = f'_seed{args._current_seed}' if hasattr(args, '_current_seed') and getattr(args, 'ensemble', False) else ''
    save_path = os.path.join(output_dir, f'model_{args.subject}{seed_suffix}.pth')

    # Load pretrained model
    print(f"Loading pretrained: {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location='cuda', weights_only=False)
    model = EEGTransformer(**MODEL_CONFIG).cuda()
    model.load_state_dict(ckpt['model_state_dict'])
    pretrained_norm_mean = ckpt['norm_mean']
    pretrained_norm_std = ckpt['norm_std']
    print(f"Pretrained norm: mean={pretrained_norm_mean:.6e}, std={pretrained_norm_std:.6e}")

    # Load C data (converted to Volts)
    train_data, train_labels, test_data, test_labels = load_C_data(subject=args.subject)
    print(f"C data (Volts): train={train_data.shape}, test={test_data.shape}")

    # Euclidean Alignment on C data (train+test combined for stable covariance)
    ea_matrix = None
    if args.ea:
        all_C = np.concatenate([train_data, test_data], axis=0)
        ea_matrix = compute_ea_matrix(all_C)
        train_data = apply_ea(train_data, ea_matrix)
        test_data = apply_ea(test_data, ea_matrix)
        print(f"  EA applied to C data ({all_C.shape[0]} trials)")

    # Use pretrained norm stats to align input distribution with pretrained model
    pretrained_norm = (pretrained_norm_mean, pretrained_norm_std) if args.use_pretrained_norm else None

    result = train_loop(
        model, train_data, train_labels, test_data, test_labels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_ratio=args.val_ratio, number_aug=3, number_seg=8,
        save_path=save_path, freeze_strategy=args.freeze, l1_lambda=0,
        pretrained_norm=pretrained_norm, ea_matrix=ea_matrix,
        label_smoothing=args.label_smoothing, cosine_lr=args.cosine_lr,
        diff_lr=args.diff_lr, online_aug=args.online_aug,
    )

    # Re-save checkpoint with µV-domain norm stats for realtime_inference.py
    final_ckpt = torch.load(save_path, weights_only=False)
    norm_mean_V = final_ckpt['norm_mean']
    norm_std_V = final_ckpt['norm_std']
    final_ckpt['norm_mean'] = np.float64(norm_mean_V * 1e6)  # V → µV for inference
    final_ckpt['norm_std'] = np.float64(norm_std_V * 1e6)
    torch.save(final_ckpt, save_path)
    print(f"Checkpoint saved: {save_path}")

    result['save_path'] = save_path

    return result


# ─── Mode: Baseline ──────────────────────────────────────────────────────────

def baseline(args):
    """Train C from scratch (same hyperparams as finetune, for fair comparison)."""
    print("=" * 60)
    print("  BASELINE: Train from scratch on custom data")
    print("=" * 60)

    ea_suffix = '_ea' if args.ea else ''
    output_dir = f"{args.output_dir}{ea_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'model_{args.subject}.pth')

    # Load C data (converted to Volts for consistency)
    train_data, train_labels, test_data, test_labels = load_C_data(subject=args.subject)
    print(f"C data (Volts): train={train_data.shape}, test={test_data.shape}")

    # Euclidean Alignment
    ea_matrix = None
    if args.ea:
        all_C = np.concatenate([train_data, test_data], axis=0)
        ea_matrix = compute_ea_matrix(all_C)
        train_data = apply_ea(train_data, ea_matrix)
        test_data = apply_ea(test_data, ea_matrix)
        print(f"  EA applied to C data ({all_C.shape[0]} trials)")

    model = EEGTransformer(**MODEL_CONFIG).cuda()

    result = train_loop(
        model, train_data, train_labels, test_data, test_labels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_ratio=args.val_ratio, number_aug=3, number_seg=8,
        save_path=save_path, freeze_strategy='none', l1_lambda=1e-4,
        ea_matrix=ea_matrix,
    )

    # Re-save with µV-domain norm stats
    final_ckpt = torch.load(save_path, weights_only=False)
    norm_mean_V = final_ckpt['norm_mean']
    norm_std_V = final_ckpt['norm_std']
    final_ckpt['norm_mean'] = np.float64(norm_mean_V * 1e6)
    final_ckpt['norm_std'] = np.float64(norm_std_V * 1e6)
    torch.save(final_ckpt, save_path)

    return result


# ─── Multi-seed Runner ───────────────────────────────────────────────────────

def run_multi_seed(func, args, seeds):
    """Run a training function with multiple seeds and report statistics."""
    results = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        args._current_seed = seed
        print(f"\n{'─'*40} Seed {seed} {'─'*40}")
        r = func(args)
        if 'accuracy' in r:
            results.append(r)
            print(f"  → Seed {seed}: acc={r['accuracy']*100:.2f}%  kappa={r['kappa']*100:.2f}%")

    if results:
        accs = [r['accuracy']*100 for r in results]
        kappas = [r['kappa']*100 for r in results]
        print(f"\n{'='*60}")
        print(f"  SUMMARY ({len(results)} seeds)")
        print(f"  Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%  "
              f"(min={min(accs):.0f}%, max={max(accs):.0f}%)")
        print(f"  Kappa:    {np.mean(kappas):.2f}% ± {np.std(kappas):.2f}%")
        print(f"{'='*60}")

    # Ensemble evaluation: load all seed checkpoints, average softmax on test set
    if getattr(args, 'ensemble', False) and len(results) >= 2 and args.mode == 'finetune':
        _evaluate_ensemble(args, seeds, results)

    return results


def _evaluate_ensemble(args, seeds, results):
    """Evaluate ensemble of multi-seed models on test set."""
    import torch

    # Load C test data
    _, _, test_data, test_labels = load_C_data(subject=args.subject)
    if args.ea:
        train_data, _, _, _ = load_C_data(subject=args.subject)
        all_C = np.concatenate([train_data, test_data], axis=0)
        ea_matrix = compute_ea_matrix(all_C)
        test_data = apply_ea(test_data, ea_matrix)

    test_data = np.expand_dims(test_data, axis=1)  # (N, 1, 3, 1000)

    ea_suffix = '_ea' if args.ea else ''
    output_dir = f"{args.output_dir}_freeze_{args.freeze}{ea_suffix}"

    # Collect softmax outputs from each model
    all_probs = []
    for seed in seeds:
        ckpt_path = os.path.join(output_dir, f'model_{args.subject}_seed{seed}.pth')
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, weights_only=False)
        model = ckpt['model'].cuda()
        model.eval()
        norm_mean = ckpt['norm_mean'] * 1e-6  # µV back to V for test data
        norm_std = ckpt['norm_std'] * 1e-6

        test_norm = (test_data - norm_mean) / norm_std
        test_t = torch.from_numpy(test_norm).float().cuda()

        with torch.no_grad():
            outputs_list = []
            for i in range(0, len(test_t), 72):
                _, out = model(test_t[i:i+72])
                outputs_list.append(out.softmax(dim=1))
            probs = torch.cat(outputs_list).cpu().numpy()
        all_probs.append(probs)

    if len(all_probs) >= 2:
        avg_probs = np.mean(all_probs, axis=0)
        y_pred = np.argmax(avg_probs, axis=1)
        y_true = test_labels - 1
        acc, prec, rec, f1, kappa = calMetrics(y_true, y_pred)
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE ({len(all_probs)} models)")
        print(f"  Accuracy: {acc*100:.2f}%  Kappa: {kappa*100:.2f}%")
        print(f"{'='*60}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ECTNet Transfer Learning')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed evaluation (e.g. 0,1,2,3,4)')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Pretrain
    p_pre = subparsers.add_parser('pretrain', help='Pretrain on all B subjects')
    p_pre.add_argument('--epochs', type=int, default=1000)
    p_pre.add_argument('--lr', type=float, default=0.001)
    p_pre.add_argument('--batch-size', type=int, default=72)
    p_pre.add_argument('--val-ratio', type=float, default=0.1)
    p_pre.add_argument('--output-dir', default='pretrained_B')
    p_pre.add_argument('--ea', action='store_true', help='Apply Euclidean Alignment per subject')

    # Finetune
    p_ft = subparsers.add_parser('finetune', help='Fine-tune pretrained model on C data')
    p_ft.add_argument('--pretrained', required=True, help='Path to pretrained checkpoint')
    p_ft.add_argument('--freeze', choices=['none', 'cnn', 'cnn+transformer4'], default='cnn')
    p_ft.add_argument('--epochs', type=int, default=200)
    p_ft.add_argument('--lr', type=float, default=0.001)
    p_ft.add_argument('--use-pretrained-norm', action='store_true',
                       help='Use pretrained B normalization stats instead of recomputing from C data')
    p_ft.add_argument('--batch-size', type=int, default=72)
    p_ft.add_argument('--val-ratio', type=float, default=0.15)
    p_ft.add_argument('--subject', type=int, default=1)
    p_ft.add_argument('--output-dir', default='transfer_C')
    p_ft.add_argument('--ea', action='store_true', help='Apply Euclidean Alignment')
    p_ft.add_argument('--ensemble', action='store_true', help='Save per-seed checkpoints for ensemble')
    p_ft.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing (e.g. 0.1)')
    p_ft.add_argument('--cosine-lr', action='store_true', help='Cosine annealing LR schedule')
    p_ft.add_argument('--diff-lr', action='store_true', help='Differential LR: CNN*0.1, Trans*0.3, Head*1.0')
    p_ft.add_argument('--online-aug', action='store_true', help='Online augmentation: noise + scaling + time shift')

    # Baseline
    p_bl = subparsers.add_parser('baseline', help='Train C from scratch (comparison)')
    p_bl.add_argument('--epochs', type=int, default=1000)
    p_bl.add_argument('--lr', type=float, default=0.001)
    p_bl.add_argument('--batch-size', type=int, default=72)
    p_bl.add_argument('--val-ratio', type=float, default=0.3)
    p_bl.add_argument('--subject', type=int, default=1)
    p_bl.add_argument('--output-dir', default='baseline_C')
    p_bl.add_argument('--ea', action='store_true', help='Apply Euclidean Alignment')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
        func = {'pretrain': pretrain, 'finetune': finetune, 'baseline': baseline}[args.mode]
        run_multi_seed(func, args, seeds)
    else:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        if args.mode == 'pretrain':
            pretrain(args)
        elif args.mode == 'finetune':
            finetune(args)
        elif args.mode == 'baseline':
            baseline(args)


if __name__ == '__main__':
    main()
