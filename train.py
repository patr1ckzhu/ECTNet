"""
CTNet Training Script

Subject-specific and cross-subject (LOSO) training for EEG motor imagery classification.
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
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
cudnn.benchmark = True
cudnn.deterministic = False

from model import EEGTransformer
from utils import calMetrics, calculatePerClass, numberClassChannel, load_data_evaluate


class ExP():
    def __init__(self, nsub, data_dir, result_name,
                 epochs=2000,
                 number_aug=2,
                 number_seg=8,
                 gpus=[0],
                 evaluate_mode = 'subject-dependent',
                 heads=4,
                 emb_size=40,
                 depth=6,
                 dataset_type='A',
                 eeg1_f1 = 20,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.3,
                 flatten_eeg1 = 600,
                 validate_ratio = 0.2,
                 learning_rate = 0.001,
                 batch_size = 72,       # each batch of raw train dataset, real training batchsize =  batch_size * (1 + N_AUG) for additional data augmentation.
                 l1_lambda = 1e-4,      # L1 regularization weight for channel attention sparsity
                 ):

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
        self.heads=heads
        self.emb_size=emb_size
        self.depth=depth
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio

        self.l1_lambda = l1_lambda
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.number_class, self.number_channel = numberClassChannel(self.dataset_type)
        self.model = EEGTransformer(
             heads=self.heads,
             emb_size=self.emb_size,
             depth=self.depth,
            database_type=self.dataset_type,
            eeg1_f1=eeg1_f1,
            eeg1_D=eeg1_D,
            eeg1_kernel_size=eeg1_kernel_size,
            eeg1_pooling_size1 = eeg1_pooling_size1,
            eeg1_pooling_size2 = eeg1_pooling_size2,
            eeg1_dropout_rate = eeg1_dropout_rate,
            eeg1_number_channel = self.number_channel,
            flatten_eeg1 = flatten_eeg1,
            ).cuda()
        #self.model = nn.DataParallel(self.model, device_ids=gpus)
        self.model = self.model.cuda()
        self.model_filename = self.result_name + '/model_{}.pth'.format(self.nSub)

    # Segmentation and Reconstruction (S&R) data augmentation (vectorized)
    def interaug(self, timg, label):
        n_per_class = self.number_augmentation * int(self.batch_size / self.number_class)
        seg_len = 1000 // self.number_seg
        aug_data = []
        aug_label = []

        for cls in range(self.number_class):
            cls_data = timg[label == cls + 1]  # (n_cls, 1, ch, 1000)
            n_cls = cls_data.shape[0]

            # Generate all random indices at once: (n_per_class, n_seg)
            rand_idx = np.random.randint(0, n_cls, (n_per_class, self.number_seg))

            # Build augmented samples by gathering random segments
            tmp = np.empty((n_per_class, 1, self.number_channel, 1000), dtype=cls_data.dtype)
            for s in range(self.number_seg):
                start = s * seg_len
                end = start + seg_len
                tmp[:, :, :, start:end] = cls_data[rand_idx[:, s], :, :, start:end]

            aug_data.append(tmp)
            aug_label.append(np.full(n_per_class, cls + 1))

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[shuffle]
        aug_label = aug_label[shuffle]

        return (torch.from_numpy(aug_data).float().cuda(),
                torch.from_numpy(aug_label - 1).long().cuda())



    def get_source_data(self):
        (self.train_data,    # (batch, channel, length)
         self.train_label,
         self.test_data,
         self.test_label) = load_data_evaluate(self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode)

        self.train_data = np.expand_dims(self.train_data, axis=1)  # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]  # (288, 1, 22, 1000)
        self.allLabel = self.allLabel[shuffle_num]


        print('-'*20, "train size：", self.train_data.shape, "test size：", self.test_data.shape)
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]


        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self, val_interval=5):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Pre-move test data to GPU once
        test_data = test_data.type(self.Tensor)
        test_label = test_label.type(self.LongTensor)

        # Pre-split train/val once (fixed split instead of re-splitting per batch)
        n_total = len(dataset)
        n_val = int(self.validate_ratio * n_total)
        n_train = n_total - n_val
        perm = torch.randperm(n_total)
        train_indices = perm[:n_train]
        val_indices = perm[n_train:]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_img = img[val_indices].type(self.Tensor)
        val_label_t = label[val_indices].type(self.LongTensor)

        # Compile model for faster execution (PyTorch 2.x)
        compiled_model = torch.compile(self.model)

        best_epoch = 0
        min_loss = 100
        result_process = []
        self.dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=False, drop_last=False)

        for e in range(self.n_epochs):
            epoch_process = {'epoch': e}
            compiled_model.train()

            for i, (batch_img, batch_label) in enumerate(self.dataloader):
                batch_img = batch_img.type(self.Tensor)
                batch_label = batch_label.type(self.LongTensor)

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                batch_img = torch.cat((batch_img, aug_data))
                batch_label = torch.cat((batch_label, aug_label))

                # training model
                features, outputs = compiled_model(batch_img)
                loss = self.criterion_cls(outputs, batch_label)

                # L1 regularization on channel attention FC parameters for sparsity
                if self.l1_lambda > 0:
                    l1_loss = sum(p.abs().sum() for p in self.model.get_channel_attention_params())
                    loss = loss + self.l1_lambda * l1_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            # Validate every val_interval epochs
            if (e + 1) % val_interval == 0 or e == self.n_epochs - 1:
                compiled_model.eval()
                with torch.no_grad():
                    _, val_outputs = compiled_model(val_img)
                    val_loss = self.criterion_cls(val_outputs, val_label_t)
                    val_pred = val_outputs.argmax(dim=1)
                    val_acc = (val_pred == val_label_t).float().mean().item()

                epoch_process['val_acc'] = val_acc
                epoch_process['val_loss'] = val_loss.item()

                train_pred = outputs.argmax(dim=1)
                train_acc = (train_pred == batch_label).float().mean().item()
                epoch_process['train_acc'] = train_acc
                epoch_process['train_loss'] = loss.item()

                if min_loss > val_loss:
                    min_loss = val_loss
                    best_epoch = e
                    epoch_process['epoch'] = e
                    torch.save(self.model, self.model_filename)
                    print("{}_{} train_acc: {:.4f} train_loss: {:.6f}\tval_acc: {:.6f} val_loss: {:.7f}".format(
                        self.nSub, e, train_acc, loss.item(), val_acc, val_loss.item()))

            result_process.append(epoch_process)

        # load model for test
        self.model.eval()
        self.model = torch.load(self.model_filename, weights_only=False).cuda()
        outputs_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_dataloader):
                img_test = Variable(img.type(self.Tensor)).cuda()

                # test model
                features, outputs = self.model(img_test)
                val_pred = torch.max(outputs, 1)[1]
                outputs_list.append(outputs)
        outputs = torch.cat(outputs_list)
        y_pred = torch.max(outputs, 1)[1]


        test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

        print("epoch: ", best_epoch, '\tThe test accuracy is:', test_acc)

        # Extract channel attention weights from test data
        ca_weights_list = []
        with torch.no_grad():
            for i, (img, _) in enumerate(self.test_dataloader):
                img = img.type(self.Tensor).cuda()
                self.model(img)
                w = self.model.get_channel_attention_weights()
                if w is not None:
                    ca_weights_list.append(w.cpu())
        if ca_weights_list:
            ca_weights = torch.cat(ca_weights_list).mean(dim=0).numpy()  # (n_channels,)
            ca_path = self.result_name + '/channel_attention_sub{}.npy'.format(self.nSub)
            np.save(ca_path, ca_weights)
            print("Channel attention weights (sub {}): {}".format(self.nSub, ca_weights))

        df_process = pd.DataFrame(result_process)

        return test_acc, test_label, y_pred, df_process, best_epoch


def main(dirs,
         evaluate_mode = 'subject-dependent', # "LOSO" or other
         heads=8,             # heads of MHA
         emb_size=48,         # token embding dim
         depth=3,             # Transformer encoder depth
         dataset_type='A',    # A->'BCI IV2a', B->'BCI IV2b'
         eeg1_f1=20,          # features of temporal conv
         eeg1_kernel_size=64, # kernel size of temporal conv
         eeg1_D=2,            # depth-wise conv
         eeg1_pooling_size1=8,# p1
         eeg1_pooling_size2=8,# p2
         eeg1_dropout_rate=0.3,
         flatten_eeg1=600,
         validate_ratio = 0.2,
         l1_lambda = 1e-4,
         ):

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    result_write_metric = ExcelWriter(dirs+"/result_metric.xlsx")

    result_metric_dict = {}
    y_true_pred_dict = { }

    process_write = ExcelWriter(dirs+"/process_train.xlsx")
    pred_true_write = ExcelWriter(dirs+"/pred_true.xlsx")
    subjects_result = []
    best_epochs = []

    for i in range(N_SUBJECT):

        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2024)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        index_round =0
        print('Subject %d' % (i+1))
        exp = ExP(i + 1, DATA_DIR, dirs, EPOCHS, N_AUG, N_SEG, gpus,
                  evaluate_mode = evaluate_mode,
                  heads=heads,
                  emb_size=emb_size,
                  depth=depth,
                  dataset_type=dataset_type,
                  eeg1_f1 = eeg1_f1,
                  eeg1_kernel_size = eeg1_kernel_size,
                  eeg1_D = eeg1_D,
                  eeg1_pooling_size1 = eeg1_pooling_size1,
                  eeg1_pooling_size2 = eeg1_pooling_size2,
                  eeg1_dropout_rate = eeg1_dropout_rate,
                  flatten_eeg1 = flatten_eeg1,
                  validate_ratio = validate_ratio,
                  l1_lambda = l1_lambda,
                  )

        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i+1))
        y_true_pred_dict[i] = df_pred_true

        accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {'accuray': accuracy*100,
                          'precision': precison*100,
                          'recall': recall*100,
                          'f1': f1*100,
                          'kappa': kappa*100
                          }
        subjects_result.append(subject_result)
        df_process.to_excel(process_write, sheet_name=str(i+1))
        best_epochs.append(best_epoch)

        print(' THE BEST ACCURACY IS ' + str(testAcc) + "\tkappa is " + str(kappa) )


        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

        df_result = pd.DataFrame(subjects_result)
    process_write.close()
    pred_true_write.close()


    print('**The average Best accuracy is: ' + str(df_result['accuray'].mean()) + "kappa is: " + str(df_result['kappa'].mean()) + "\n" )
    print("best epochs: ", best_epochs)
    result_metric_dict = df_result

    mean = df_result.mean(axis=0)
    mean.name = 'mean'
    std = df_result.std(axis=0)
    std.name = 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])

    df_result.to_excel(result_write_metric, index=False)
    print('-'*9, ' all result ', '-'*9)
    print(df_result)

    print("*"*40)

    result_write_metric.close()


    return result_metric_dict

if __name__ == "__main__":
    #----------------------------------------
    DATA_DIR = r'./mymat_raw/'
    EVALUATE_MODE = 'LOSO-No' # leaving one subject out subject-dependent  subject-indenpedent

    N_SUBJECT = 9       # BCI
    N_AUG = 3           # data augmentation times for generating artificial training data set
    N_SEG = 8           # segmentation times for S&R

    EPOCHS = 1000
    EMB_DIM = 16
    HEADS = 2
    DEPTH = 6
    TYPE = 'A'
    validate_ratio = 0.3 # split raw train dataset into real train dataset and validate dataset

    EEGNet1_F1 = 8
    EEGNet1_KERNEL_SIZE=64
    EEGNet1_D=2
    EEGNet1_POOL_SIZE1 = 8
    EEGNet1_POOL_SIZE2 = 8
    FLATTEN_EEGNet1 = 240
    L1_LAMBDA = 1e-4        # L1 regularization for channel attention sparsity

    if EVALUATE_MODE!='LOSO':
        EEGNet1_DROPOUT_RATE = 0.5
    else:
        EEGNet1_DROPOUT_RATE = 0.25


    number_class, number_channel = numberClassChannel(TYPE)
    RESULT_NAME = "{}_heads_{}_depth_{}".format(TYPE, HEADS, DEPTH)

    sModel = EEGTransformer(
        heads=HEADS,
        emb_size=EMB_DIM,
        depth=DEPTH,
        database_type=TYPE,
        eeg1_f1=EEGNet1_F1,
        eeg1_D=EEGNet1_D,
        eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
        eeg1_pooling_size1 = EEGNet1_POOL_SIZE1,
        eeg1_pooling_size2 = EEGNet1_POOL_SIZE2,
        eeg1_dropout_rate = EEGNet1_DROPOUT_RATE,
        eeg1_number_channel = number_channel,
        flatten_eeg1 = FLATTEN_EEGNet1,
        ).cuda()
    summary(sModel, (1, number_channel, 1000))

    print(time.asctime(time.localtime(time.time())))

    result = main(RESULT_NAME,
                    evaluate_mode = EVALUATE_MODE,
                    heads=HEADS,
                    emb_size=EMB_DIM,
                    depth=DEPTH,
                    dataset_type=TYPE,
                    eeg1_f1 = EEGNet1_F1,
                    eeg1_kernel_size = EEGNet1_KERNEL_SIZE,
                    eeg1_D = EEGNet1_D,
                    eeg1_pooling_size1 = EEGNet1_POOL_SIZE1,
                    eeg1_pooling_size2 = EEGNet1_POOL_SIZE2,
                    eeg1_dropout_rate = EEGNet1_DROPOUT_RATE,
                    flatten_eeg1 = FLATTEN_EEGNet1,
                    validate_ratio = validate_ratio,
                    l1_lambda = L1_LAMBDA,
                  )
    print(time.asctime(time.localtime(time.time())))
