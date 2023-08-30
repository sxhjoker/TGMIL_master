from __future__ import print_function

import argparse
import csv
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils_6 import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

def to_csv(results_path):
    f = open(results_path, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['slide_id','y_true', 'y_pred'])
    f.close()


def write_csv(results,results_path):
    with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        for i in results:
            wf.writerow(i)
def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_precision = []
    all_val_precision = []
    all_test_racall = []
    all_val_racall = []
    all_test_F1 = []
    all_val_F1 = []
    folds = np.arange(start, end)#k折交叉验证
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results_dict_val,val_pre_results,results_dict_test,test_pre_results, test_auc, val_auc, test_acc, val_acc,val_precision, val_racall, val_F1, test_precision, test_racall, test_F1 = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_precision.append(test_precision)
        all_val_precision.append(val_precision)
        all_test_racall.append(test_racall)
        all_val_racall.append(val_racall)
        all_test_F1.append(test_F1)
        all_val_F1.append(val_F1)
        #write results to pkl
        filename_val = os.path.join(args.results_dir, 'split_{}_val_results.pkl'.format(i))
        save_pkl(filename_val, results_dict_val)
        filename_test = os.path.join(args.results_dir, 'split_{}_test_results.pkl'.format(i))
        save_pkl(filename_test, results_dict_test)
        pre_results_path = os.path.join(args.results_dir, 'split_{}_val_results.csv'.format(i))
        to_csv(pre_results_path)
        write_csv(np.array(val_pre_results), pre_results_path)
        test_pre_results_path = os.path.join(args.results_dir, 'split_{}_test_results.csv'.format(i))
        to_csv(test_pre_results_path)
        write_csv(np.array(test_pre_results), test_pre_results_path)
        print(i,'val_auc',val_auc,'val_acc',val_acc,'val_precision',val_precision,'val_racall',val_racall,'val_F1',val_F1)


    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc,
        'test_precision': all_test_precision, 'val_precision' : all_val_precision,
        'test_racall': all_test_racall, 'val_racall' : all_val_racall,
        'test_F1': all_test_F1, 'val_F1': all_val_F1})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', default='./Kidney_512_123_TCGA_FEATURES_DIRECTORY',type=str,
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5 5e-4)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results/TGMIL', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd','lookahead_radam'], default='lookahead_radam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['TGMIL'], default='TGMIL',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', default='task_2_tumor_subtyping_CLAM_50',type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'],default='task_2_tumor_subtyping')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default='svm',
                     help='instance-level clustering loss function (default: None) tcga:svm')
parser.add_argument('--subtyping', action='store_true', default=True,
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

parser.add_argument('--num_classes', default=3, type=int, help='Number of output classes [2]')




args = parser.parse_args()
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = './CAMELYON16_Step_3.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = './TCGA_Step_3.csv',
                            data_dir= os.path.join(args.data_root_dir, 'Three_subtypes_featurs'),
                            shuffle = False,
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'KICH':0, 'KIRC':1, 'KIRP':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_mb']:
        assert args.subtyping 
        
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('./TCGA_splits_5', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('./TCGA_splits_5', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


#104保存全局平均池化，最大池化结果和图结果，但是还解释不了得到的矩阵代表什么意思。
# 还尝试解决生成热图图像加载不全的问题，可能与图像的下采样，层数相关