"""Quick evaluation of 22ch baseline models to extract per-subject accuracy."""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
from utils import load_data_evaluate, calMetrics

DATA_DIR = './mymat_raw/'
MODEL_DIR = 'compare_9sub_baseline'

for sub in range(1, 10):
    train_data, train_label, test_data, test_label = load_data_evaluate(DATA_DIR, 'A', sub, mode_evaluate='LOSO-No')

    train_data_exp = np.expand_dims(train_data, axis=1)
    target_mean = np.mean(train_data_exp)
    target_std = np.std(train_data_exp)

    test_data = np.expand_dims(test_data, axis=1)
    test_label_t = np.transpose(test_label)
    test_data = (test_data - target_mean) / target_std
    test_label_flat = test_label_t[0]

    test_tensor = torch.from_numpy(test_data).float().cuda()
    test_label_tensor = torch.from_numpy(test_label_flat - 1).long().cuda()

    model = torch.load(f'{MODEL_DIR}/model_{sub}.pth', weights_only=False).cuda()
    model.eval()

    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_label_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=72, shuffle=False)

    outputs_list = []
    with torch.no_grad():
        for img, _ in test_loader:
            _, out = model(img.cuda())
            outputs_list.append(out)

    outputs = torch.cat(outputs_list)
    y_pred = torch.max(outputs, 1)[1]

    acc, prec, rec, f1, kappa = calMetrics(
        test_label_tensor.cpu().numpy().astype(int),
        y_pred.cpu().numpy().astype(int)
    )
    print(f'S{sub}: acc={acc*100:.2f}%, kappa={kappa*100:.2f}%')

    del model
    torch.cuda.empty_cache()
