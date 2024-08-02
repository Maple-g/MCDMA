from CLS.marginmatch.marginmatch_utils import AUMCalculator, consistency_loss, Get_Scalar, replace_threshold_examples
from copy import deepcopy
from CLS.Process import dataload, Eval, generate_batches, Generate_features, data_split
import torch.optim as optim
from CLS.model import *
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample, shuffle
from collections import Counter
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from alive_progress import alive_bar
import random
import sys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from CLS.dalib.modules.domain_discriminator import DomainDiscriminator
from CLS.dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from CLS.dalib.adaptation.mcc import MinimumClassConfusionLoss
from CLS.dalib.modules.masking import Masking
from CLS.dalib.modules.teacher import EMATeacher
from CLS.common.utils.meter import AverageMeter
OverSampler = RandomOverSampler()
# # %%
sys.path.append('../')
init_mmd_params = {'MMD_LAMBDA': 0.25, 'MMD_GAMMA': 5}
mmd_params = init_mmd_params.copy()
seed = 2024
loss = nn.BCEWithLogitsLoss()
Over = RandomOverSampler()
Under = RandomUnderSampler()

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(epoch):
    i = 0
    it = 0
    model.train()
    Average_loss = 0

    S_graph_data, S_edge_data, S_labels = S_data
    T_graph_data, T_edge_data, T_labels = T_data
    U_graph_data, U_edge_data, _ = U_data
    batch_epoch = len(U_graph_data) // batch_size
    Source_generator = generate_batches(S_graph_data, S_edge_data, S_labels, batch_size)
    Target_labels_generator = generate_batches(T_graph_data, T_edge_data, T_labels, batch_size)
    Target_Unlabels_generator = generate_batches(U_graph_data, U_edge_data, _, batch_size)

    for _ in range(batch_epoch):
        S_graph_files, S_edge_files, S_labels, S_idx = next(Source_generator)
        T_graph_files, T_edge_files, T_labels, T_idx = next(Target_labels_generator)
        U_graph_files, U_edge_files, U_labels, U_idx = next(Target_Unlabels_generator)

        S_input, S_graph_sizes = Generate_features(S_graph_files, S_edge_files)
        T_input, t_graph_sizes = Generate_features(T_graph_files, T_edge_files)
        U_input, u_graph_sizes = Generate_features(U_graph_files, U_edge_files)

        S_labels = torch.FloatTensor(S_labels).cuda()
        T_labels = torch.FloatTensor(T_labels).cuda()

        features_t = Feature_extract(T_input)
        features_s = Feature_extract(S_input)
        features_u = Feature_extract(U_input)

        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        optimizer_f.zero_grad()
        Teacher.update_weights(model, epoch * 1000)
        pseudo_label_t, pseudo_prob_t = Teacher(features_t, t_graph_sizes)

        y_s, f_s = model(features_s, S_graph_sizes)
        Pred_t, _ = model(features_t, t_graph_sizes)
        y_t, f_t = model(features_u, u_graph_sizes)

        T_sup_loss = loss(Pred_t, T_labels)
        S_sup_loss = loss(y_s, S_labels)

        masking = Masking(
            block_size=64,
            ratio=0.4,
        ).cuda()
        x_u_masked = masking(features_u)
        x_t_masked = masking(features_t)
        masking_f = Masking(
            block_size=64,
            ratio=0.2,
        ).cuda()
        x_ulb_w = masking_f(features_u)
        logits_x_ulb_w, _ = model(x_ulb_w, u_graph_sizes)
        masking_S = Masking(
            block_size=64,
            ratio=0.6,
        ).cuda()
        x_ulb_s = masking_S(features_u)
        logits_x_ulb_s, _ = model(x_ulb_s, u_graph_sizes)

        ulb_dset = features_u

        aum_calculator = AUMCalculator(0.997, int(1), len(ulb_dset), 1 / 100)

        threshold_example_cutoff = ((2 ** 20) * 9) // 10

        if it < threshold_example_cutoff and it >= 80000:
            replace_threshold_examples(ulb_dset, aum_calculator)
        else:
            aum_calculator.switch_threshold_examples(set())
            # ulb_dset.switch_threshold_examples(set())


        if it % 20 == 0 and it > 80000:
            if 0.9977 < it and it < threshold_example_cutoff:
                aum_threshold = aum_calculator.retrieve_threshold()
        elif it == 30:
            aum_threshold = 0
            break
        elif it < 30:
            aum_threshold = None

        threshold_mask = True
        t_fn = Get_Scalar(0.5)
        p_fn = Get_Scalar(0.95)
        T = t_fn(0)
        p_cutoff = p_fn(0)
        selected_label_confidence = torch.ones(
            (len(ulb_dset),), dtype=torch.long, ) * -1
        selected_label_confidence = selected_label_confidence.cuda()
        classwise_acc_confidence = torch.zeros((10,)).cuda()

        pseudo_counter_confidence = Counter(
            selected_label_confidence.tolist())

        thresh_warmup = True

        if max(pseudo_counter_confidence.values()) < len(ulb_dset):
            if thresh_warmup:
                for x in range(10):
                    classwise_acc_confidence[x] = pseudo_counter_confidence[x] / \
                                                  max(pseudo_counter_confidence.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter_confidence)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for x in range(10):
                    classwise_acc_confidence[x] = pseudo_counter_confidence[x] / \
                                                  max(wo_negative_one.values())
            classwise_acc_confidence[1] = 0

        T_idx = torch.FloatTensor(T_idx).cuda()


        logits_x_ulb_w = torch.sigmoid(logits_x_ulb_w)
        logits_x_ulb_s = torch.sigmoid(logits_x_ulb_s)
        lambda_u = 10

        (unsup_loss, mask_confidence, mask_aum, select_confidence, _, pseudo_lb, p_model, conf_acc, both_acc, aum_acc,
         mask_sum) = consistency_loss(logits_x_ulb_s, logits_x_ulb_w,
                                                       classwise_acc_confidence,
                                                       0,
                                                       None,
                                                       T_idx,
                                                       aum_calculator,
                                                       0,
                                                       aum_threshold,
                                                       1,
                                                       threshold_mask,
                                                       'ce', T, p_cutoff,
                                                       use_hard_labels=True,
                                                       use_DA=False,
                                                       labels=False)

        if T_idx[select_confidence == 1].nelement() != 0:
            selected_label_confidence[T_idx[select_confidence == 1]
            ] = pseudo_lb[select_confidence == 1]


        y_u_masked, _ = model(x_u_masked, u_graph_sizes)
        y_t_masked, _ = model(x_t_masked, t_graph_sizes)


        if Teacher.pseudo_label_weight is not None:
            ce = loss(y_u_masked, pseudo_label_t) + loss(y_t_masked, T_labels)
            masking_loss_value = torch.mean(pseudo_prob_t * ce) + torch.mean(T_labels * ce)
        else:
            masking_loss_value = loss(y_u_masked, pseudo_label_t) + loss(y_t_masked, T_labels)

        transfer_loss = domain_adv(y_s, f_s, y_t, f_t) + S_sup_loss


        mask_loss = masking_loss_value + mcc_loss(y_t) + mcc_loss(Pred_t)
        l = T_sup_loss + transfer_loss + mask_loss + lambda_u * unsup_loss

        Average_loss += l

        l.backward()
        optimizer_ad.step()
        optimizer.step()
        optimizer_f.step()

        scheduler.step()
        scheduler_f.step()
        scheduler_ad.step()

        print("loss {}".format(l.item()))

    Average_loss = Average_loss/batch_epoch

    return Average_loss


def test(Epoch_stop):
    with torch.no_grad():
        model.eval()
        Input, graph_sizes = Generate_features(U_graph_data, U_edge_data)
        labels = torch.FloatTensor(U_labels).float().cuda()
        feature = Feature_extract(Input)
        outputs, _ = model(feature, graph_sizes)
        outputs = torch.sigmoid(outputs)

        output = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        y_preds = output[:] > 0.45

        Eval(Train, Test, Epoch_stop, labels, y_preds, output, save_path)


class EarlyStopping:
    """早停机制的实现"""
    def __init__(self, patience=10, verbose=True, delta=0.1, save_on_patience=True):
        self.patience = patience
        self.verbose = verbose
        self.save_on_patience = save_on_patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # 第一次遇到的最佳分数时立即保存
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                # if self.save_on_patience:
                #     # 仅在耐心耗尽时保存模型
                #     self.save_checkpoint(val_loss, model)
        else:
            self.best_score = score
            self.counter = 0
            # if not self.save_on_patience:
            #     # 如果不是仅在耐心耗尽时保存，则在每次改进时保存
            #     self.save_checkpoint(val_loss, model)
    def save_checkpoint(self, val_loss, model):
        '''保存模型当验证损失减少时'''
        path = 'Model_Save/' + Test + '/'

        # 检查路径是否存在，如果不存在，则创建它
        if not os.path.exists(path):
            os.makedirs(path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')

        # 保存模型到指定路径
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss

save_path = 'Pseudo_labels_Results' + '.csv'
def start_run():
    with alive_bar(len(range(epoch))) as bar:

        Epoch_stop = 0

        print('----------------- First Train -----------------')
        for j in range(epoch):
            bar()
            l = train(epoch)
            early_stopping(l, model)
            if early_stopping.early_stop:
                print("early stopping")
                Epoch_stop = j
                break

        test(Epoch_stop)

epoch = 70
patience = 10
Multi_Source = {'EF': 'DE',
                'UC': 'EF',
                'BN': 'RE',
                'SE': 'RE',
                'DE': 'RE',
                'TP': 'OF'
                }
# train_list = ['EF', 'RE', 'DE', 'EF', 'OF', 'UC', 'SE']
Test_list = ['TP']
# for num_train_samples in Samples_list:
for Test in Test_list:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Define the StratifiedKFold object
    fold_data = []
    test_data, test_labels = dataload(Test)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    for train_index, test_index in skf.split(test_data, test_labels):

        New_train_data, New_test_data = test_data[test_index], test_data[train_index]
        New_train_labels, New_test_labels = test_labels[test_index], test_labels[train_index]
        New_train_data, New_train_labels = OverSampler.fit_resample(X=New_train_data, y=New_train_labels)

        New_train_data, New_train_labels = shuffle(New_train_data, New_train_labels)
        T_graph_data, T_edge_data, T_labels, U_graph_data, U_edge_data, U_labels = data_split(New_train_data,
                                                                                              New_train_labels,
                                                                                              New_test_data,
                                                                                              New_test_labels)
        T_data = (T_graph_data, T_edge_data, T_labels)
        U_data = (U_graph_data, U_edge_data, U_labels)

        # for Train in train_list:

        Train = Multi_Source.get(Test)
        S_data, S_labels = dataload(Train)
        S_data, S_labels = Under.fit_resample(X=S_data, y=S_labels)
        S_data, S_labels = shuffle(S_data, S_labels)
        S_graph_data = [data[0] for data in S_data]
        S_edge_data = [data[1] for data in S_data]
        S_data = (S_graph_data, S_edge_data, S_labels)
        Feature_extract = Feature(classNum=1,
                                  dropout_rate=0.5,
                                  nfeat=100,
                                  nhid=11,
                                  nclass=1,
                                  n_layers=9,
                                  k=200,
                                  head=2,
                                  features_length=100
                                  )
        model = Predictor(classNum=1,
                          dropout_rate=0.5,
                          nhid=11,
                          nclass=1,
                          n_layers=9,
                          k=200,
                          features_length=100
                          )
        domain_discri = DomainDiscriminator(
            128, hidden_size=128)
        classifier_feature_dim = 20000
        domain_adv = ConditionalDomainAdversarialLoss(
            domain_discri, entropy_conditioning=False,
            num_classes=1, features_dim=classifier_feature_dim, randomized=False,
            randomized_dim=1024
        )
        Teacher = EMATeacher(model, alpha=0.999, pseudo_label_weight=2.0)
        mcc_loss = MinimumClassConfusionLoss(temperature=2.0)
        batch_size = 10
        if torch.cuda.is_available():
            model.cuda()
        Feature_extract.cuda()
        Teacher.cuda()
        domain_adv.cuda()
        domain_discri.cuda()

        optimizer_ad = optim.Adam(domain_discri.parameters(), lr=0.1)
        scheduler_ad = optim.lr_scheduler.CyclicLR(optimizer_ad, base_lr=0.0001, max_lr=0.0004,
                                                   cycle_momentum=False)

        optimizer = optim.Adam(model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.0004, cycle_momentum=False)

        optimizer_f = optim.Adam(Feature_extract.parameters(), lr=0.1)
        scheduler_f = optim.lr_scheduler.CyclicLR(optimizer_f, base_lr=0.0001, max_lr=0.0004,
                                                  cycle_momentum=False)
        early_stopping = EarlyStopping(patience, verbose=True)

        wf = './'
        start_run()

       #






