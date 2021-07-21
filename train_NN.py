import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import numpy as np
import pandas as pd
import os

from tqdm import tqdm

import math

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import pickle

import warnings

## loggerの設定
from logging import debug, getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO
logger =getLogger("logger")
logger.setLevel(DEBUG)
## StreamHandlerの設定
handler1 = StreamHandler()
handler1.setLevel(DEBUG)
handler1.setFormatter(Formatter("%(asctime)s: %(message)s"))
## FileHandlerの設定
config_filename = "train_NN"
handler2 = FileHandler(filename=f'./logs/{config_filename}.log')
handler2.setLevel(DEBUG)
handler2.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler1)
logger.addHandler(handler2)

# 設定まとめ
'''
fold_num : kfoldの分割数
seed : 乱数を使用する時のシード値
params : lgbmモデルのパラメータ
target_name : 目的変数のカラム名
ID_name : submissionのID名
Debug : Trueで、1foldでbreak
'''

class Config:
    def __init__(self):
        self.fold_num:int = 5
        self.q_splits:int = 20
        self.seed:int=777
        self.params = {
            "learning_rate": 0.01,
            "random_seed": 42,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
        }
        self.Debug = False
        
CFG = Config()

train_path = "./input/train"
test_path = "./input/test"
sub_path = "./input/submission"

# モデル本体
class NNModel(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout_rate):
        super(NNModel, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(num_features, hidden_size)
        self.batch_norm_mid = nn.BatchNorm1d(hidden_size)
        self.dropout_mid = nn.Dropout(dropout_rate)
        self.dense_mid = nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_size, num_targets)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        x = self.batch_norm_mid(x)
        x = self.dropout_mid(x)
        x = F.relu(self.dense_mid(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, in_df, out_df):
        self.in_df = in_df
        self.out_df = out_df
    def __getitem__(self, index):
        # インデックス index のサンプルが要求されたときに返す処理を実装
        input_data = torch.tensor(self.in_df.iloc[index].values.astype(np.float32), dtype=torch.float)
        output_data = torch.tensor(self.out_df.iloc[index], dtype=torch.float)
        return input_data, output_data
    def __len__(self):
        return len(self.in_df)

from time import time
from statistics import mean

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

df_train = pd.read_feather("./output/train.feather")
df_public = pd.read_feather("./output/X_pubric.feather")
df_private = pd.read_feather("./output/X_private.feather")
# 特徴量と目的変数に分離
train = df_train.drop(["death", "date"], axis=1)
y_train_all = df_train.death

logger.debug("loaded dataset")

qcut_target = pd.qcut(y_train_all, CFG.q_splits, labels=False, duplicates='drop')

device = "cuda:1" if torch.cuda.is_available() else "cpu"
epoch = 50
criterion = RMSELoss()
folds = StratifiedKFold(n_splits=CFG.fold_num, shuffle=True, random_state=CFG.seed).split(np.arange(train.shape[0]), qcut_target)

preds_NN_pub = []
preds_NN_pri = []

for fold, (trn_idx, val_idx) in enumerate(folds):
    logger.debug("fold:{} start".format(fold))
    # dataset 作成
    X_train = train.loc[trn_idx, :]
    X_valid = train.loc[val_idx, :]
    y_train = y_train_all.loc[trn_idx]
    y_valid = y_train_all.loc[val_idx]
    
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_valid, y_valid)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 2048, shuffle = True, num_workers = 4, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 2048, shuffle = False, num_workers = 4, drop_last = False)
    
    model_me = NNModel(num_features = X_train.shape[1], num_targets = 1, hidden_size = 128, dropout_rate = 0.1)
    model_me = model_me.to(device)
    
    params = [{"params": model_me.parameters()}]
    
    optimizer = Adam(params, lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)
    best_val = 1000000.
    
    for _ in range(epoch):
        start_time = time()
        train_loss = []
        val_loss = []
            
        train_out_list = []
        train_target_list = []
            
        val_out_list = []
        val_target_list = []
            
                
        # train
        # モードを切り替える
        model_me.train()
        
        for data in train_loader:
            inp, out_ans = data
            inp = inp.to(device)
            out_ans = out_ans.to(device)
            
            # 誤差の初期化を行う
            optimizer.zero_grad()
                
            out_dat = model_me(inp)
            out_dat.squeeze_(1)
            
            loss = criterion(out_dat, out_ans)
            loss.backward()
                
            train_loss.append(criterion(out_dat, out_ans).item())
            optimizer.step()
            scheduler.step()
        
        # val
        # モードを切り替える
        model_me.eval()
        # 勾配の計算をしない
        with torch.no_grad():
            for data in val_loader:
                inp, out_ans = data
                inp = inp.to(device)
                out_ans = out_ans.to(device)
                
                out_dat = model_me(inp)
                out_dat.squeeze_(1)
                
                loss = torch.sqrt(criterion(out_dat, out_ans))
                val_loss.append(criterion(out_dat, out_ans).item())
        
        logger.debug(f'## epoch {_+1}, {(time()-start_time)/60:.2f} min  train_loss: {math.sqrt(mean(train_loss)):.5f}, val_loss: {math.sqrt(mean(val_loss)):.5f}')
       
        if best_val > math.sqrt(mean(val_loss)):
            best_val = math.sqrt(mean(val_loss))
            torch.save(model_me.state_dict(), f'best_{fold}.pth')
                                 
    model_me.load_state_dict(torch.load(f'best_{fold}.pth'))
    
    test_set = Dataset(df_public.drop("date", axis=1), y_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False, 
                                             num_workers = 4, drop_last = False)
    ans_this_cv = []
    for data in test_loader:
        inp, out_ans = data
        out_dat = model_me(inp.to(device))
        out_dat.squeeze_(1)
        
        ans_this_cv.extend(out_dat.tolist())
    
    assert len(ans_this_cv) == df_public.shape[0]
    preds_NN_pub.append(ans_this_cv)
    
    test_set = Dataset(df_private.drop("date", axis=1), y_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False, 
                                             num_workers = 4, drop_last = False)
    ans_this_cv = []
    for data in test_loader:
        inp, out_ans = data
        out_dat = model_me(inp.to(device))
        out_dat.squeeze_(1)
        
        ans_this_cv.extend(out_dat.tolist())
    
    assert len(ans_this_cv) == df_private.shape[0]
    preds_NN_pri.append(ans_this_cv)

preds_NN_pub = np.mean(preds_NN_pub, axis=0)
preds_NN_pri = np.mean(preds_NN_pri, axis=0)

sub_pub_NN = pd.read_csv(os.path.join(sub_path, "submission_public.csv"))
sub_pri_NN = pd.read_csv(os.path.join(sub_path, "submission_private.csv"))

# スマートな方法が思いつかなかったのでfor文です

def public(x):
    return preds_NN_pub[x.index]

def private(x):
    return preds_NN_pri[x.index]
    
tmp = df_public.groupby("id").apply(public)
for i in range(len(tmp)):
    sub_pub_NN.iloc[i, 1:] = tmp[i]
    
tmp = df_private.groupby("id").apply(private)
for i in range(len(tmp)):
    sub_pri_NN.iloc[i, 1:] = tmp[i]

final_path = './output/'

sub_pub_NN.to_csv(
    final_path+"sub_pub_NN.csv",
    index = False
)
sub_pri_NN.to_csv(
    final_path+"sub_pri_NN.csv",
    index = False
)