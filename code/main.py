import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import jieba
import pandas as pd
import gensim.models.word2vec as w2v
from sklearn.model_selection import train_test_split
import word2id
import torchtext

parser=argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-num', type=int, default=383132, help='number of onehot words [default: 383132]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=16, help='number of each kind of kernel')
parser.add_argument('-class-num', type=int, default=10, help='number of output class')
parser.add_argument('-kernel-sizes', type=str, default=[3,4,5], help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

args = parser.parse_args()
class textCNN(nn.Module):
    
    def __init__(self, args):
        super(textCNN, self).__init__()
        self.args = args
        
        V = args.embed_num #已知词的数量：test集可能出现新词
        D = args.embed_dim
        C = args.class_num #output: 分类的类型数
        Ci = 1
        Co = args.kernel_num #out_channel,每个卷积核可以将句子理解为某一种新的特征，通道数+1
        Ks = args.kernel_sizes
        #只有一层CNN 
        #embedding--->CONV2D--->RELU--->MAXPOOL--->DROPOUT
        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)


    def forward(self, x):
        x = self.embed(x)  # (N, W, D) W:分词后每个句子的长度
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks) 每个Ks对应一个16通道的卷积层，卷完的结果是一维，因为kernel_size的width和embed_dim相同
        														  #W是否应该是W-K+1?
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks) #len(Ks)长度的x i是其中每一个relu出的结果 
        #input ,kernel_size,*args(optional)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
#data_format: {"fact_cut": "公诉 机关 指控 。", "accu": 110, "law": 32, "time": 4, "term_cate": 2, "term": 9}

def data_fetch(filename1,filename2):
    LABEL = torchtext.data.Field(sequential=False,use_vocab=False)
    TEXT = torchtext.data.Field(sequential=True,fix_length=500)
    infile=open(filename1,'r',encoding="utf-8")
    X_plaintext,X,Y_accu,Y_law,Y_time,Y_term_cate,Y_term=[],[],[],[],[],[]
    while True:
        line=infile.readline()
        if line:
            
            data=json.loads(line)
            try:
                X_plaintext.append(data['fact_cut'])
                Y_accu.append(data['accu'])
                Y_law.append(data['law'])
                Y_time.append(data['time'])
                Y_term_cate.append(data['term_cate'])
                Y_term.append(data['term'])
                tv_datafields = [#("id", None), # 我们不会需要id，所以我们传入的filed是None
                    ("term", LABEL), ("law", LABEL),
                    ("fact_cut", TEXT), ("time", LABEL),
                    ("accu", LABEL), ("term_cate", LABEL)
                ]
                trn, tst= torchtext.data.TabularDataset.splits(
                    path="../data/useful/", # 数据存放的根目录
                    train='new_data_train_cuted.csv', test="new_data_test_cuted.csv",
                    format='csv',
                    skip_header=True, # 如果你的csv有表头, 确保这个表头不会作为数据处理
                    fields=tv_datafields)
                train_iter,  test_iter = torchtext.data.BucketIterator.splits(
                    (train, test), sort_key=lambda x: len(x.fact_cut),
                    batch_sizes=(args.batch_size, 1), device=5)

            except:
                print("Line misses item!") 
                continue


        else:
            break

if __name__=='__main__':
    data_fetch()


#model modification
#CNN后添加attention层，层参数


##data format
#x_train,x_test,y_train,y_test
#{‘label’: ‘xx’, ‘text’:‘xxxxx’} json or txt文件格式, by torchtext
#text:分词+ (语法树构建，每句话中的词对抽取)
#label:law_articles, accusation, terms