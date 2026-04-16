"""Generate A/B comparison table from baseline and channel attention results."""
import numpy as np

# Baseline results (from baseline_full_output.txt)
b_acc = [90.62, 69.44, 92.01, 82.64, 73.26, 67.71, 88.54, 83.33, 87.15]
b_kap = [87.50, 59.26, 89.35, 76.85, 64.35, 56.94, 84.72, 77.78, 82.87]
b_ep  = [953, 872, 923, 930, 698, 953, 971, 969, 971]
b_tm  = [351.1, 351.8, 350.8, 350.4, 353.1, 354.1, 353.3, 353.5, 353.3]

# Channel attention results (from ca_full_output.txt)
c_acc = [90.28, 70.14, 92.71, 79.51, 75.69, 62.85, 92.36, 82.29, 85.76]
c_kap = [87.04, 60.19, 90.28, 72.69, 67.59, 50.46, 89.81, 76.39, 81.02]
c_ep  = [975, 863, 902, 987, 937, 917, 995, 801, 988]
c_tm  = [375.9, 373.3, 372.8, 373.0, 374.1, 373.6, 374.9, 373.8, 375.6]

print("=" * 100)
print("  CONTROLLED A/B COMPARISON: BASELINE vs CHANNEL ATTENTION")
print("  (BCI Competition IV-2a, 22 channels, 4-class, Fixed Seed=42)")
print("=" * 100)
print()

header = "  Sub | Base Acc  | CA Acc    | dAcc    | Base Kap  | CA Kap    | dKap    | B Ep  | C Ep  | B Time | C Time"
print(header)
print("-" * 100)

for i in range(9):
    da = c_acc[i] - b_acc[i]
    dk = c_kap[i] - b_kap[i]
    das = "+" if da >= 0 else ""
    dks = "+" if dk >= 0 else ""
    print("  S%d  | %7.2f%%  | %7.2f%%  | %s%.2f%% | %7.2f%%  | %7.2f%%  | %s%.2f%% | %5d | %5d | %5.1fs | %5.1fs" % (
        i+1, b_acc[i], c_acc[i], das, da, b_kap[i], c_kap[i], dks, dk, b_ep[i], c_ep[i], b_tm[i], c_tm[i]))

print("-" * 100)

bm_a = np.mean(b_acc)
bs_a = np.std(b_acc)
cm_a = np.mean(c_acc)
cs_a = np.std(c_acc)
bm_k = np.mean(b_kap)
bs_k = np.std(b_kap)
cm_k = np.mean(c_kap)
cs_k = np.std(c_kap)
da = cm_a - bm_a
dk = cm_k - bm_k

print("  Mean| %7.2f%%  | %7.2f%%  | %s%.2f%% | %7.2f%%  | %7.2f%%  | %s%.2f%% |       |       | %5.1fs | %5.1fs" % (
    bm_a, cm_a, "+" if da >= 0 else "", da, bm_k, cm_k, "+" if dk >= 0 else "", dk, np.mean(b_tm), np.mean(c_tm)))
print("  Std | %7.2f%%  | %7.2f%%  |         | %7.2f%%  | %7.2f%%  |         |       |       |        |" % (
    bs_a, cs_a, bs_k, cs_k))
print("=" * 100)

print()
print("PER-SUBJECT DELTA ANALYSIS:")
deltas_acc = [c - b for c, b in zip(c_acc, b_acc)]
deltas_kap = [c - b for c, b in zip(c_kap, b_kap)]
improved = sum(1 for d in deltas_acc if d > 0)
worsened = sum(1 for d in deltas_acc if d < 0)
tied = sum(1 for d in deltas_acc if d == 0)
print("  Accuracy improved: %d/9 subjects" % improved)
print("  Accuracy worsened: %d/9 subjects" % worsened)
print("  Best improvement:  S%d (%+.2f%%)" % (np.argmax(deltas_acc) + 1, max(deltas_acc)))
print("  Worst regression:  S%d (%+.2f%%)" % (np.argmin(deltas_acc) + 1, min(deltas_acc)))

print()
print("CHANNEL ATTENTION WEIGHT ANALYSIS (across 9 subjects):")
ch_names = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
try:
    ws = [np.load('compare_9sub_ca/channel_attention_sub%d.npy' % s) for s in range(1, 10)]
    mean_w = np.mean(ws, axis=0)
    top_idx = np.argsort(mean_w)[::-1]
    print("  Ranked channels (high to low attention weight):")
    for rank, idx in enumerate(top_idx):
        marker = " <-- HIGH" if rank < 5 else (" <-- LOW" if rank >= 17 else "")
        print("    %2d. %-4s (ch%2d): %.4f%s" % (rank + 1, ch_names[idx], idx + 1, mean_w[idx], marker))
    print()
    print("  Weight range: [%.4f, %.4f], spread = %.4f" % (mean_w.min(), mean_w.max(), mean_w.max() - mean_w.min()))
    print("  Mean weight: %.4f +/- %.4f" % (np.mean(mean_w), np.std(mean_w)))
except Exception as e:
    print("  Could not load .npy files: %s" % e)

print()
print("CONCLUSION:")
print("  Baseline mean accuracy:          %.2f%% +/- %.2f%%" % (bm_a, bs_a))
print("  Channel attention mean accuracy:  %.2f%% +/- %.2f%%" % (cm_a, cs_a))
print("  Delta:                            %+.2f%%" % da)
print("  Baseline mean kappa:             %.2f%% +/- %.2f%%" % (bm_k, bs_k))
print("  Channel attention mean kappa:     %.2f%% +/- %.2f%%" % (cm_k, cs_k))
print("  Delta:                            %+.2f%%" % dk)
print()
if abs(da) < 1.0:
    print("  The channel attention mechanism shows NEGLIGIBLE difference (< 1%% accuracy change)")
    print("  from the baseline. The two models perform comparably on this dataset/configuration.")
elif da > 0:
    print("  The channel attention mechanism shows a POSITIVE improvement of +%.2f%%." % da)
else:
    print("  The channel attention mechanism shows a NEGATIVE regression of %.2f%%." % da)
