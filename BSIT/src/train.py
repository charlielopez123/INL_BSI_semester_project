import argparse
import os
import torch
import random
import numpy as np
from exp.exp_classification import Exp_Classification
from exp.exp_bsi_pretrain import Exp_BSI_PRETRAIN



def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():


    parser = argparse.ArgumentParser(description='Hyperparameters Settings')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--plot_graph', type=int, default=0, help='plotting')
    parser.add_argument('--streaming', type=int, default=0, help='stream the detection data')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='CNNLSTM', help='model name, options: [CNNLSTM]')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_batch', type=bool, default=True, help = "if use batched data")
    parser.add_argument('--dataset', type=str, default='Epilepsy', help='model name, options: [Epilepsy,HAR,HAR70,HandMovementDirection,FingerMovements,SpokenArabicDigits,MotorImagery,fingerreg]')    
    parser.add_argument('--task', type=str, default="Classification", choices = ["Classification","Detection","SSL","SSLDetection","SSLEval","SSLJoint","SSLJointDetection","Regression","SSLVQ"],help = "task type")
    parser.add_argument('--label_informed',action = 'store_true', help = 'use label informed structure')
    parser.add_argument('--cluster',action = 'store_true', help = 'use clustering loss')
    parser.add_argument('--contrastive',action = 'store_true', help = 'use contrastive loss')
    parser.add_argument('--supcon',action = 'store_true', help = 'use supervised contrastive loss')


    # data loader
    parser.add_argument('--checkpoints', type=str, default='/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--adj_mat_dir', type=str, default='src/data/adj_mx_3d.pkl', help='location of adj matrix')
    parser.add_argument('--max_clip_length', type=int, default=11, help='location of adj matrix')
    parser.add_argument('--pre_ictal_length', type=int, default=1, help='pre ictal length')
    parser.add_argument('--use_fft', action='store_true', help='use fourier transform')
    parser.add_argument('--clip_length', type=int, default=1, help='clip length for dataset')
    parser.add_argument('--clip_stride', type=int, default=1, help='clip length for dataset')
    parser.add_argument('--low_seizure', type=int, default=1, help='lower bound of seizure')
    parser.add_argument('--high_seizure', type=int, default=5, help='upper bound of seizure')
    parser.add_argument('--dataset_dir', type=str, default='/dataset/', help='location of datasets')

    # model define
    parser.add_argument('--n_classes', type=int, default=4, help='class number')
    parser.add_argument('--seq_length', type=int, default=2500, help='input sequence length')
    parser.add_argument('--num_nodes', type=int, default=19, help='number of nodes')
    parser.add_argument('--embed_dim', type=int, default=64, help='token dim')
    parser.add_argument('--input_dim', type=int, default=100, help='input channel')
    parser.add_argument('--output_dim', type=int, default=100, help='ouput channel')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden units')
    parser.add_argument('--num_t_pints', type=int, default=750, help='time sequence')
    parser.add_argument('--num_heads', type=int, default=4, help='attention heads')
    parser.add_argument('--num_decheads', type=int, default=4, help='attention decoder heads')
    parser.add_argument('--num_layers', type=int, default=2, help='attention layers')
    parser.add_argument('--num_declayers', type=int, default=2, help='attention decoder layers')
    parser.add_argument('--num_patches', type=int, default=2, help='token patches')
    parser.add_argument('--num_cut', type=int, default=3, help='patches cut')
    parser.add_argument('--sample_len', type=int, default=256, help='pred length')

    parser.add_argument('--kernel_size', type=int, default=3, help='conv1d kernel')
    parser.add_argument('--stride', type=int, default=1, help='conv1d stride')
    parser.add_argument("--augmentations",type=str, default="flip,shuffle,frequency,jitter,mask,drop", help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="2,4,8",
        help="a list of patch len used in Medformer",
    )

    parser.add_argument('--preprocess', action='store_true', help='use mnefilter')
    parser.add_argument('--n_embed', type=int, default=64, help='codec size')
    parser.add_argument('--wavelet_scale', type=int, default=5, help='wavelet scales')
    parser.add_argument('--use_wavelet', action='store_true', help='use wavelet')
    parser.add_argument('--use_bandpass', action='store_true', help='use bandpass')
    parser.add_argument('--use_cwt', action='store_true', help='use cwt') #debug
    parser.add_argument('--decimation', action='store_true', help='decimation of cwt') #debug
    parser.add_argument('--n_time_downsampled',  type=int, default=590, help='downsampling n_time points for cwt') #debug
    parser.add_argument('--wavelet', type=str, default='cmor1.0-1.0', help='wavelet type for cwt') #debug
    parser.add_argument('--decim_filter', type=str, default='iir', help='alias filter used for decimation') #debug
    parser.add_argument('--averaging', action = 'store_true',  help='averaging of cwt')# debug
    parser.add_argument('--pca', action = 'store_true',  help='pca for feature selection')# debug
    parser.add_argument('--pca_features', type=int, default=384,  help='number of features for feature selection')# debug
    parser.add_argument('--forward_selection', action = 'store_true', help='forward_selection') # debug
    parser.add_argument('--feature_selection_wrapper', action = 'store_true', help='feature selection using wrapper method') # debug
    parser.add_argument('--feature_selection_features', type=int, default=384,  help='number of features for feature selection')# debug





    parser.add_argument('--filter_type', type=str, default="laplacian", help='filter type')
    parser.add_argument('--use_curriculum_learning', action='store_true', help='use curriculum learning')
    parser.add_argument('--max_diffusion_step', type=int, default=2, help='Maximum Diffusion Step')
    parser.add_argument('--cl_decay_steps', type=int, default=3000, help='Scheduled sampling decay steps.')
    parser.add_argument('--dcgru_activation', type=str, default='tanh', help='dropout rate')
    parser.add_argument('--num_rnn_layers', type=int, default=3, help='input channel')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Maximum gradient norm for gradient clipping.')

    # GTS model
    parser.add_argument('--temperature', type=float, default=0.5, help='gumbel temp')
    parser.add_argument('--embedding_dim', type=int, default=64, help='embedding_dim')
    parser.add_argument('--fix_graph', type=int, default=0, help='keep graph fixed')
    parser.add_argument('--regularizer', type=int, default=0, help='regularized graph')
    parser.add_argument('--base_graph', type=int, default=0, help='use basic graph')
    parser.add_argument('--only_weight', type=int, default=0, help='only change weights')
    parser.add_argument('--only_edge', type=int, default=0, help='only change edges')
    parser.add_argument('--fully_connected', type=int, default=0, help='fully connected A')
    parser.add_argument('--sample_times', type=int, default=5, help='graph sampling times')

    #pretrain model
    parser.add_argument('--fine_tune', action='store_true', help='use pretrained model')
    parser.add_argument('--pretrain_model', type=str, default="DCRNN_Pred", help='pretrain model')
    parser.add_argument('--pretrain_model_path', type=str, default="DCRNN_SSL1fft_DCRNN_test", help='pretrain model path')
    parser.add_argument('--pretrained_num_rnn_layers', type=int, default=3, help='pretrain model rnn layers')
    parser.add_argument('--pt_class', type=int, default=4, help='pretrain model fc class')
    parser.add_argument('--free_graph', action='store_true', help='free graph layers')
    parser.add_argument('--only_graph', action='store_true', help='only transfer graph layers')
    parser.add_argument('--prob_mt', action='store_true', help='not use time series to learn graph')
    parser.add_argument('--linear_probing', action='store_true', help='freeze the encoder')
    parser.add_argument('--probing_epochs', type=int, default=30, help='epoch to freeze encoder')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='masking channels')



    # others
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_activation', type=bool, default=True, help='use_activation')
    parser.add_argument('--validate_dataset', type=str, default='test', help='which dataset to calculate loss')

    #auxiliary loss
    parser.add_argument('--supcon_weight_class', type=float, default=0.0005, help='weight for supcon_loss')
    parser.add_argument('--supcon_weight_detection', type=float, default=0.001, help='weight for supcon_loss')
    parser.add_argument('--reconstruction_weight', type=float, default=0.1, help='weight for cos_loss')
    parser.add_argument('--simsiam_reconstruction_weight', type=float, default=0.5, help='weight for cos_loss')
    parser.add_argument('--simsiam_prediction_weight', type=float, default=0.25, help='weight for cos_loss')
    parser.add_argument('--cluster_momentum', type=float, default=0.05, help='weight for cos_loss')
    parser.add_argument('--cluster_attract_weight', type=float, default=0.25, help='weight for cos_loss')
    parser.add_argument('--cluster_repel_weight_dist', type=float, default=0.125, help='weight for cos_loss')
    parser.add_argument('--cluster_repel_weight_angle', type=float, default=0.125, help='weight for cos_loss')
    parser.add_argument('--cluster_prediction_weight', type=float, default=0.5, help='weight for cos_loss')
    parser.add_argument('--cluster_margin', type=float, default=0.25, help='margin of cluster')
    parser.add_argument('--w_main_task', type=float, default=0.5, help='w_main_task')
    parser.add_argument('--w_auxiliary_task', type=float, default=0.5, help='w_auxiliary_task')
    parser.add_argument('--drop_task_epoch', type=int, default=20, help='when drop auxiliary task')
    parser.add_argument('--aug_variance', type=float, default=0.5, help='variance of augmentation')
    parser.add_argument('--augmentation', action = 'store_true', help='data augmentation')
    parser.add_argument('--noise_clustering', action = 'store_true', help='noise supervised clustering')
    parser.add_argument('--input_augmenting', action = 'store_true', help='augment input signal')
    parser.add_argument('--etf', action = 'store_true', help='use etf classifier')
    parser.add_argument('--noise_samples', type=int, default=1, help='number of noisy samples')
    parser.add_argument('--pretrain_method', type=str, default='simsiam', help='pretrain_method')
    parser.add_argument('--scaling', action = 'store_true', help='scale data')

    # tSNE vis
    parser.add_argument('--plot_epoch', type=int, default=5, help='frequency of plotting embedding visualization')
    parser.add_argument('--ssl_split', type=int, default=5, help='partition of training sets')
    # parser.add_argument('--ssl_assess', action='store_true', help='use pre-trained model to generate representations ')


    # optimization
    parser.add_argument('--N_WORKERS', type=int, default=0, help='data loader num workers')
    parser.add_argument('--prefetch_factor', type=int, default=None, help='pre-fetch')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--n_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--early_stop', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer regularization')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=str, default='cuda:2', help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')



    args = parser.parse_args()
    # args.use_gpu = False
    # print("use_gpu: ",args.use_gpu)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    fix_seed = args.seed
    seed_everything(fix_seed)

    print('Args in experiment:')
    print(args)


    if args.task == "Classification":
        Exp = Exp_Classification
    elif args.task == "SSLEval":
        Exp = Exp_BSI_PRETRAIN
    elif args.task == "Regression":
        Exp = Exp_Regression
    elif args.task == "SSLVQ":
        Exp = Exp_BSIVQ_PRETRAIN
    else:
        Exp = Exp_Detection
    
    # Exp = Exp_GTS_Detection_Test




    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if args.forward_selection:
                exp.forward_selection()
            elif args.feature_selection_wrapper:
                exp.feature_selection_wrapper_method()
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.plot_graph:
                print('>>>>>>>plotting graph of : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test_plot(setting)

            torch.cuda.empty_cache()
    else:
        if args.plot_graph:
            
            ii = 0
            setting = '{}_{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.des, ii)
            print('>>>>>>>plotting graph of : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp = Exp(args)  # set experiments
            
            exp.test_plot(setting)
            torch.cuda.empty_cache()
        else:

            
            ii = 0
            setting = '{}_{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.des, ii)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp = Exp(args)  # set experiments
            
            exp.test(setting)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    os.nice(20)
    main()
    # cProfile.run("main()")