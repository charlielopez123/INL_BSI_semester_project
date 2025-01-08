from exp.exp_basic import Exp_Basic

from models import BSIformer,biot,Medformer,BSIformer_vq,BSIformer_rec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import datetime
import gc
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR


# from data.dataloader_test import dataloader_test
from data.data_utils import calc_data_scale
from data.dataloader import create_non_graph_loader, create_BSI_loader
import utils
import copy
import seaborn as sns

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression




warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.train_scaler = None
        self.criterion = self._select_criterion()
        self.args = args
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.embedding = None
        self.labels = None
        self.valid_embedding = None
        self.valid_labels = None
        self.scaler , self.cls_counts = calc_data_scale(args)
        self.avg_trace = None
        if args.fine_tune:
            self.args_pretrained = copy.deepcopy(args)
            # setattr(
            #         self.args_pretrained,
            #         'n_classes',
            #         6)
            self.pretrained_model = self._build_pretrained_model().to(self.device)
            pretrain_output_path = os.path.join(self.args.checkpoints, args.pretrain_model_path)
            self.pretrained_model.load_state_dict(torch.load(os.path.join(pretrain_output_path, 'checkpoint.pth'), map_location='cpu'))
            if args.model == "BSIformerVQ":
                self.model = utils.build_finetune_bsiformervq(model_new=self.model, model_pretrained=self.pretrained_model)
            elif args.model == "BSIformer_REC":
                self.model = utils.build_finetune_bsiformerrec(model_new=self.model, model_pretrained=self.pretrained_model)
            else:
                self.model = utils.build_finetune_bsiformer(model_new=self.model, model_pretrained=self.pretrained_model)
            # self.model = utils.build_finetune_lstm_model(model_new=self.model,model_pretrained=self.pretrained_model,num_rnn_layers=args.num_rnn_layers)

    
    def _build_model(self):
        model_dict = {
            'BSIformer':BSIformer,
            'ConvBSIformer':BSIformer,
            'BSIformerVQ':BSIformer_vq,
            'BSIformer_REC':BSIformer_rec,
            'BSIformerT':BSIformer,
            'biot':biot,
            'Medformer':Medformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _build_pretrained_model(self):
        model_dict = {

            'BSIformer':BSIformer,
            'BSIformerVQ':BSIformer_vq,
            'BSIformer_REC':BSIformer_rec

        }
        model = model_dict[self.args.pretrain_model].Model(self.args_pretrained).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    # def _get_data(self):
    #     dataset_train,dataset_test,dataset_vali,trainloader,testloader,valiloader = dataloader_test()
    #     return dataset_train,dataset_test,dataset_vali,trainloader,testloader,valiloader



    def _select_optimizer(self, nettype='all'):
        if nettype == 'all':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('wrong type.')
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        # if self.args.dataset != "Epilepsy":
        #     criterion = nn.CrossEntropyLoss()
        # else:
        # criterion = utils.MultiClassFocalLossWithAlpha(self.device)
        return criterion

    
    def train_epoch_batched(self, model, optimizer, train_loader):
        model.train()
        criterion = self.criterion
        loss_sup = []
        self.embedding = None
        self.labels = None
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.n_epochs)
        for batch_X, batch_y in tqdm(train_loader, total=len(train_loader)):
            # --------------- forward --------------- #
            channel = batch_X.reshape(-1,batch_X.shape[-1])
            # print("batch shape:",batch_X.shape,channel.shape)
            # meanc = torch.mean(channel,dim=0)
            # print("mean of each channel: ", meanc)
            output_y,embedding = model.forward(batch_X.float().to(self.device),train=True)

            # print("output test: ",output_y.shape,batch_y.to(self.device).shape,type(output_y),type(batch_y.to(self.device)))
            loss_batch = criterion(output_y, batch_y.squeeze().to(self.device))

            
            
            if self.embedding is None:
                self.embedding = embedding
                self.labels = batch_y.cpu().detach().numpy().squeeze()
            else:
                self.embedding = np.concatenate((self.embedding,embedding),axis=0)
                self.labels = np.concatenate((self.labels,batch_y.cpu().detach().numpy().squeeze()),axis=0)
            # ----------- Parameters update --------------- #
            optimizer.zero_grad()
            loss_batch.backward()
            nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm)
            optimizer.step()

            loss_sup.append(loss_batch.item())
        loss_sup_ = np.array(loss_sup).mean(axis=0)
        scheduler.step()

        return loss_sup_
    
    def valid_batch(self, model, valid_loader):
        model.eval()
        total_loss = 0

        criterion = self.criterion
        loss_list = []
        y_pred_all = []
        y_true_all = []
        y_prob_all = []
        self.valid_embedding = None
        self.valid_labels = None

        for batch_X, batch_y in tqdm(valid_loader, total=len(valid_loader)):
            with torch.no_grad():
                output_y,embedding = model.forward(batch_X.float().to(self.device))
            # print(output_y,batch_y)
            y_prob = F.softmax(output_y, dim=1).cpu().numpy()
                # print("y prob: ",y_prob)
            y_pred = np.argmax(y_prob, axis=1).reshape(-1)
            # y_true = batch_y
                

            loss = criterion(output_y, batch_y.squeeze().to(self.device))

            
            if self.valid_embedding is None:
                self.valid_embedding = embedding
                self.valid_labels = batch_y.numpy().squeeze()
            else:
                self.valid_embedding = np.concatenate((self.valid_embedding,embedding),axis=0)
                self.valid_labels = np.concatenate((self.valid_labels,batch_y.numpy().squeeze()),axis=0)

            # print("current loss: ",loss)
            # print(np.concatenate((np.expand_dims(y_pred,1),np.expand_dims(batch_y,1)),axis=1))
            total_loss += loss.item()
            loss_list.append(loss.item())
            # print("y shape: ",y_pred.shape)
            y_pred_all.append(y_pred)
            y_true_all.append(batch_y)
            y_prob_all.append(y_prob)

        loss_list = np.array(loss_list)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        # print("loss info: ",np.mean(loss_list),np.var(loss_list),len(loss_list),len(valid_loader))
        loss_mean = total_loss/len(loss_list)
        matrix = confusion_matrix(y_true_all,y_pred_all)
        print("confusion_matrix: ")
        print(matrix)
        matrix2 = confusion_matrix(y_true_all,y_pred_all,normalize = 'true')
        self.avg_trace = np.trace(matrix2)/matrix2.shape[0]

        scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all)

        results_list = [('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision'])]

        return results_list[1][1],loss_mean

    def feature_selection_wrapper_method(self):#debug
        print("--------------------------feature selection wrapper----------------------------------")
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=self.args.feature_selection_features)
        task_op = None
        X_train, y_train = create_BSI_loader(self.args,"train",self.scaler, shuffle=True,task_op=task_op)
        print(f"X_train.shape {X_train.shape}")
        print(f"y_train.shape {y_train.shape}")
        rfe.fit_transform(X_train, y_train)
        selected_features_rfe = np.where(rfe.support_)[0]
        print("Selected Features (RFE):", selected_features_rfe)
        print(type(selected_features_rfe))

        self.args.current_features = selected_features_rfe
        return selected_features_rfe

    def forward_selection(self):#debug
        print("--------------------------forward selection----------------------------------")
        #self.pprint("git branch name: ",self.branch_name)
        selected_features = []
        remaining_features = list(range(self.args.num_t_pints))
        print(f"forward selection current number of features: {len(selected_features)}")

        while len(selected_features) < self.args.feature_selection_features:
            
            best_feature = None
            best_score_feature = -float('inf')

            for feature in remaining_features:
                current_features = selected_features + [feature]
                self.args.current_features = current_features
                

                self.model = self._build_model().to(self.device)
                num_model = self.count_parameters(self.model)
                print('#model params:', num_model)

                if self.args.model == "BSIformer" or self.args.model == "BSIformerT":
                    _cls_num_list = torch.Tensor(self.cls_counts)
                    self.model.model.margin = torch.log(_cls_num_list / torch.sum(_cls_num_list)).to(self.device)
                    print(self.model.model.margin)
                
                optimizer = self._select_optimizer()

                freeze_flag = False

                if self.args.linear_probing:
                    freeze_flag = True
                
                task_op = None

                if "BSI" in self.args.dataset:
                    self.train_loader = create_BSI_loader(self.args,"train",self.scaler, shuffle=True,task_op=task_op)
                    print("finished train dataloader forward selection")
                    self.vali_loader = create_BSI_loader(self.args,"valid",self.scaler,shuffle=False,task_op=task_op)


                #valid_score = -float('inf')
                for epoch in range(1,5): #run for 5 epochs
                    print('Epoch f selection:', epoch, '/5')
                    print('training...')

                    if epoch > self.args.probing_epochs and freeze_flag and self.args.model == "BSIformer":
                        self.model.model.transformer.requires_grad = True
                        freeze_flag = False

                    loss_sup = self.train_epoch_batched(self.model,optimizer,self.train_loader)

                    print(loss_sup)

                    print('evaluating...')


                    valid_score, valid_loss = self.valid_batch(self.model, self.vali_loader)

                
                if valid_score > best_score_feature:
                    best_score_feature = valid_score
                    best_feature = feature
            
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        
        return selected_features


#################################################################################

    
    def train(self, settings):
        output_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        folder_path = '/home/maetz/INL/yuhan_setup/train_loss/' + settings + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(folder_path+'npys/'):
            os.makedirs(folder_path+'npys/')
        self.args.log_file = os.path.join(output_path, 'run.log')

        for var in vars(self.args):
            self.pprint('{}:{}'.format(var,vars(self.args)[var]))
        
        self.pprint("------------------------------------------------------------")
        #self.pprint("git branch name: ",self.branch_name)
        
        num_model = self.count_parameters(self.model)
        self.pprint('#model params:', num_model)

        if self.args.model == "BSIformer" or self.args.model == "BSIformerT":
            _cls_num_list = torch.Tensor(self.cls_counts)
            self.model.model.margin = torch.log(_cls_num_list / torch.sum(_cls_num_list)).to(self.device)
            print(self.model.model.margin)

        optimizer = self._select_optimizer()

        best_score = -np.inf
        best_epoch, stop_round = 0, 0

        train_score_list = []
        valid_score_list = []
        test_score_list = []

        train_loss_list = []
        valid_loss_list = []
        test_loss_list = []
        Log_ME_list = []
        SFDA_list = []
        NLEEP_list = []
        freeze_flag = False

        if self.args.linear_probing:
            freeze_flag = True

        # if self.args.dataset == "HAR":
        #     self.train_loader = create_HAR_loader(self.args,"train",shuffle=True)
        #     self.vali_loader = create_HAR_loader(self.args,"valid",shuffle=False)
        #     self.test_loader = create_HAR_loader(self.args,"test",shuffle=False)
        # elif self.args.dataset == "HAR70":
        #     self.train_loader = create_HAR70_loader(self.args,"train",shuffle=True)
        #     self.vali_loader = create_HAR70_loader(self.args,"valid",shuffle=False)
        #     self.test_loader = create_HAR70_loader(self.args,"test",shuffle=False)
        task_op = None
        if self.args.dataset == "FingerMovements":
            task_op = "Detection"
        

        
        if "BSI" in self.args.dataset:
            self.train_loader = create_BSI_loader(self.args,"train",self.scaler, shuffle=True,task_op=task_op)
            self.vali_loader = create_BSI_loader(self.args,"valid",self.scaler,shuffle=False,task_op=task_op)
            self.test_loader = create_BSI_loader(self.args,"test",self.scaler, shuffle=False,task_op=task_op)

        else:

            self.train_loader = create_non_graph_loader(self.args,"train",shuffle=True,task_op=task_op)
            self.vali_loader = create_non_graph_loader(self.args,"valid",shuffle=False,task_op=task_op)
            self.test_loader = create_non_graph_loader(self.args,"test",shuffle=False,task_op=task_op)


        for epoch in range(1,self.args.n_epochs):
            self.pprint('Epoch:', epoch)
            self.pprint('training...')

            if epoch > self.args.probing_epochs and freeze_flag and self.args.model == "BSIformer":
                # for name, param in self.model.named_parameters():
                #     # 仅对浮点类型的参数设置 requires_grad=True
                #     if torch.is_floating_point(param):
                #         param.requires_grad = True
                self.model.model.transformer.requires_grad = True
                freeze_flag = False
                

        
            loss_sup = self.train_epoch_batched(self.model,optimizer,self.train_loader)

            self.pprint(loss_sup)

            self.pprint('evaluating...')

            
            train_score, train_loss = self.valid_batch(self.model, self.train_loader)

            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting training embeddings...')
                self.plot_embedding(self.embedding,self.labels,epoch,train_score,folder_path)

            valid_score, valid_loss = self.valid_batch(self.model, self.vali_loader)
            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting valid embeddings...')
                self.plot_embedding(self.valid_embedding,self.valid_labels,epoch,train_score,folder_path+'valid_plot/')

            test_score, test_loss = self.valid_batch(self.model, self.test_loader)
            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting test embeddings...')
                self.plot_embedding(self.valid_embedding,self.valid_labels,epoch,train_score,folder_path+'test_plot/')

            train_score_list.append(train_score)
            valid_score_list.append(valid_score)
            test_score_list.append(test_score)

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)

            logme_value = self.LogME_basic(self.embedding,self.labels)
            sfda_value = self.SFDA_score(self.embedding,self.labels)
            nleep_value = self.NLEEP_score(self.embedding,self.labels)

            
            Log_ME_list.append(logme_value)
            SFDA_list.append(sfda_value)
            NLEEP_list.append(nleep_value)

            self.pprint('valid %.6f, test %.6f, logme %.6f, sfda %.6f, nleep %.6f, avg_trace %.6f' %
               (valid_score, test_score, logme_value, sfda_value, nleep_value, self.avg_trace))

            if valid_score > best_score:
                best_score = valid_score
                stop_round = 0
                best_epoch = epoch
                torch.save(self.model.state_dict(), output_path+'/checkpoint.pth')
            else:
                stop_round += 1
                if stop_round >= self.args.early_stop:
                    self.pprint('early stop')
                    break
            # adjust_learning_rate(optimizer, epoch+1, self.args)

        self.pprint('best val score:', best_score, '@', best_epoch)
        self.plot_score(train_score_list,valid_score_list,test_score_list,folder_path,self.args)
        self.plot_loss(train_loss_list,valid_loss_list,test_loss_list,folder_path,self.args)
        self.plot_SSL_metrics(Log_ME_list,SFDA_list,NLEEP_list, folder_path,self.args)
        print("train data path: ",str(folder_path))

        return self.model
    
    def test(self, settings):
        output_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        folder_path = '/home/maetz/INL/yuhan_setup/test_result/' + settings + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.args.log_file = os.path.join(output_path, 'run.log')
        
           
        if not self.args.is_training and "BSI" in self.args.dataset:
            self.test_loader = create_BSI_loader(self.args,"test",self.scaler, shuffle=False)
        elif not self.args.is_training:
            self.test_loader = create_non_graph_loader(self.args,"test",shuffle=False)


        print("model path: ",str(self.args.log_file))
        criterion = self.criterion


        self.pprint('load models...')
        self.model.load_state_dict(torch.load(os.path.join(output_path, 'checkpoint.pth'), map_location='cpu'))
        num_model = self.count_parameters(self.model)
        print('#model params:', num_model)
        y_pred_all = []
        y_true_all = []
        y_prob_all = []
        loss_list = []
        total_loss = 0
        
        self.pprint('Calculate the metrics.')
        self.model.eval()
        with torch.no_grad():
            # labels = test_data.groupby('label')
            # print(labels.count())
            # grouped = test_data.groupby('session')
            # total_sessions = len(grouped)
            # print("total length: ",len(test_data))
            # labels = test_data["label"].values
            # total_len = count_consecutive_segments(test_data,self.args.max_clip_length)

            for batch_X, batch_y in tqdm(self.test_loader, total=len(self.test_loader)):
                output_y,_ = self.model.forward(batch_X.float().to(self.device))
                # print("y prob: ",output_y)
                # print("batch_y shape: ",batch_y.shape)
                y_prob = F.softmax(output_y, dim=1).cpu().numpy()
                # print("y prob: ",y_prob)
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)
                y_true = batch_y
                # print(y_pred,y_true)
                # print(torch.unique(batch_y))
                if self.args.dataset != "Epilepsy":
                    loss = criterion(output_y, y_true.squeeze().to(self.device))
                else:
                    loss = criterion(output_y, y_true.to(self.device))
            
                total_loss += loss.item()
                loss_list.append(loss.item())
                # print("y_pred shape: ",y_pred.shape)

                y_pred_all.append(y_pred)
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)

            y_pred_all = np.concatenate(y_pred_all, axis=0)
            y_true_all = np.concatenate(y_true_all, axis=0)
            y_prob_all = np.concatenate(y_prob_all, axis=0)
        print("pred len y: ",y_pred_all.shape[0])
        unique_elements, counts = np.unique(y_pred_all, return_counts=True)
        element_counts = dict(zip(unique_elements, counts))
        for element, count in element_counts.items():
            print(f"class {element} occur {count} times")
        
        self.pprint("true len y: ",y_true_all.shape[0])
        unique_elements, counts = np.unique(y_true_all, return_counts=True)
        element_counts = dict(zip(unique_elements, counts))
        for element, count in element_counts.items():
            self.pprint(f"class {element} occur {count} times")

        loss_list = np.array(loss_list)
        loss_mean = total_loss/len(loss_list)
        matrix = confusion_matrix(y_true_all,y_pred_all,normalize = 'true')
        print("confusion_matrix: ")
        print(matrix.round(2))
        self.draw_confusion_matrix(matrix,folder_path)
        self.draw_prediction_plot(y_true_all,y_prob_all,folder_path)
        self.avg_trace = np.trace(matrix)/matrix.shape[0]

        scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all)

        results_list = [('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision'])]

        self.pprint('the result of the test set:')
        self.pprint('acc:{}, F1:{}, recall:{}, precision: {}, cross_entropy: {}, avg_trace: {}'.format(results_list[0][1],results_list[1][1],results_list[2][1],results_list[3][1],loss_mean,self.avg_trace))
        

        return 
    
    def pprint(self, *text):
        # print with UTC+8 time
        time_ = '['+str(datetime.datetime.utcnow() + 
                        datetime.timedelta(hours=8))[:19]+'] -'
        print(time_, *text, flush=True)
        if self.args.log_file is None:
            return
        with open(self.args.log_file, 'a') as f:
            print(time_, *text, flush=True, file=f)
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def draw_confusion_matrix(self, matrix, folder_path):

        conf_matrix = matrix
        if self.args.n_classes == 6:
            labels = ['idle','shoulder_flexion','elbow_extension','wrist_pronation','hand_open','hand_close']
        elif self.args.n_classes == 8:
            labels = ['idle','shoulder_flexion','elbow_extension','wrist_pronation','wrist_supination','hand_open','hand_close','elbow_flexion']
        elif self.args.n_classes == 5:
            labels = ['shoulder_flexion','elbow_extension','wrist_pronation','hand_open','hand_close']
        elif self.args.n_classes == 4:
            labels = ['idle','elbow_extension','wrist_pronation','hand_open']
        elif self.args.n_classes == 3:
            labels = ['elbow_extension','wrist_pronation','hand_open']

        
        plt.figure(figsize=(18, 12))
        ax = sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                        xticklabels=labels, yticklabels=labels)

        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Normalized Confusion Matrix")
        ax.set_aspect("equal")

        plt.savefig(folder_path+'conf_matrix.jpg',dpi=300)

    def draw_prediction_plot(self, y_true, y_prob, folder_path):
        # 假设的样本数量和四分类的预测概率（y_prob）
        n_samples = y_true.shape[0]
        # y_true = np.random.randint(0, 4, size=n_samples)  # 真实标签，随机生成
        # y_prob = np.random.rand(n_samples, 4)  # 每个样本的预测概率
        y_prob /= y_prob.sum(axis=1, keepdims=True)  # 确保每个样本的概率和为 1

        # 设置颜色列表，每个类别一个颜色
        colors = ['r', 'g', 'b', 'y']  # 四个类别的颜色
        class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

        # 创建图形
        plt.figure(figsize=(12, 6))

        # 计算每个样本的最大预测值和对应的类别
        max_class_probabilities = np.argmax(y_prob, axis=1)  # 对于每个样本，找到最大预测概率对应的类别
        max_probabilities = np.max(y_prob, axis=1)  # 最高的预测概率

        # 绘制背景色，基于真实标签的区域
        for i in range(4):
            indices = np.where(y_true == i)[0]  # 找到真实标签为 i 的所有样本的索引
            for idx in indices:
                # 为每个样本区域添加背景色，使用较高的透明度
                plt.axvspan(idx - 0.5, idx + 0.5, color=colors[i], alpha=0.2)

        # 绘制其他类别的灰色概率曲线
        for i in range(4):
            plt.plot(range(n_samples), y_prob[:, i], color='gray', alpha=0.5, lw=1)  # 灰色曲线，较细

        # 绘制最高预测概率的曲线，曲线的颜色根据类别变化
        for i in range(1, n_samples):
            if max_class_probabilities[i] == max_class_probabilities[i - 1]:
                # 如果预测类别不变，继续绘制当前类别的颜色
                plt.plot([i-1, i], [max_probabilities[i-1], max_probabilities[i]], color=colors[max_class_probabilities[i]], lw=1)
            else:
                # 如果预测类别发生变化，开始新的颜色
                plt.plot([i-1, i], [max_probabilities[i-1], max_probabilities[i]], color=colors[max_class_probabilities[i]], lw=1)


        # 设置图形的 x 轴和 y 轴
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction Probability')
        plt.title('Prediction Probability Curves with True Class Background')

        # 显示图例，并确保显示正确颜色
        # 创建一个空的 label 列表，添加对应的线条颜色到 legend
        lines = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(4)]
        plt.legend(lines, class_labels, loc='upper right')

        # 显示图形
        plt.tight_layout()
        plt.savefig(folder_path+'pred_plot.jpg',dpi=300)
        

    