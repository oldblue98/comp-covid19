import numpy as np
import pandas as pd
import os
import math

from sklearn.preprocessing import StandardScaler

def main():
    train_path = "./input/train"
    test_path = "./input/test"
    sub_path = "./input/submission"

    train_death = pd.read_csv(os.path.join(train_path, "train_death.csv"))
    train_infection = pd.read_csv(os.path.join(train_path, "train_infection.csv"))
    public_infection = pd.read_csv(os.path.join(test_path, "public_infection.csv"))
    private_infection = pd.read_csv(os.path.join(test_path, "private_infection.csv"))

    city_info = pd.read_csv(os.path.join(train_path, "city_info.csv"))
    sample_submission = pd.read_csv(os.path.join(sub_path, "submission_public.csv"))

    # 実行に約90秒ほどかかります

    '''
    データ作成

    特徴量
    -過去21日分のinfection情報
    -id(地域ごとに異なる)
    -city情報(populationなど)

    trainデータのみ
    -正解データ(death)
    '''

    # 1/22/20から何日後からのデータを使用するか
    Days_later = 220

    # 過去21日分の情報を付加
    Lags = 28

    # train, public, privateの日数
    n_train = train_infection.shape[1] - Days_later -1
    n_public = public_infection.shape[1] -1
    n_private = private_infection.shape[1] -1

    # カラム名 Lag
    cols = ["inf_Lags_{}".format(i) for i in range(Lags)] + ["death"]

    all_infection = pd.concat([train_infection.iloc[:, 1:], public_infection.iloc[:, 1:], private_infection.iloc[:, 1:]], axis=1)

    def make_dataset(id):
        # infectionデータ(1/22/20から404日分)のDays_later日目以降、をデータとして採用
        #　内包表記で0~Lags日だけシフトさせ、axis=1方向に結合
        df = pd.concat([all_infection.iloc[id, Days_later:].shift(j) for j in range(Lags)], axis=1)
        df = pd.concat([df, train_death.iloc[id, Days_later+1:].astype(int)], axis=1)
        # カラム名付け
        df = df.set_axis(cols, axis='columns')
        
        #感染者数の移動平均なども追加する
        df['inf_avg7'] = all_infection.iloc[id, Days_later:].shift(1).rolling(window=7).mean()
        df['inf_avg14'] = all_infection.iloc[id, Days_later:].shift(1).rolling(window=14).mean()
        df['inf_avg21'] = all_infection.iloc[id, Days_later:].shift(1).rolling(window=21).mean()
        
        df['inf_max7'] = all_infection.iloc[id, Days_later:].shift(1).rolling(window=7).max()
        df['inf_max14'] = all_infection.iloc[id, Days_later:].shift(1).rolling(window=14).max()
        df['inf_max21'] = all_infection.iloc[id, Days_later:].shift(1).rolling(window=21).max()
        
        #その他メタデータ付加
        df["id"] = id
        df["Population"] = city_info.loc[id, "Population"].astype(int)
        df["Province_State"] = city_info.loc[id, "Province_State"]
        df = df.reset_index().rename(columns={'index': 'date'})
        return df

    def concat_dataset():
        #　内包表記でaxis=0方向（行方向）に結合
        df_train = pd.concat([make_dataset(id) for id in range(len(train_infection))], axis=0)
        return df_train

    # データ生成
    all_df = concat_dataset()

    # Province_Stateをダミー変数化
    all_df = pd.concat([all_df, pd.get_dummies(all_df.pop("Province_State"))], axis=1)

    # dateを日時用データに変換
    all_df["date"] = pd.to_datetime(all_df["date"])
    # 月情報、曜日情報を追加
    all_df["month"] = all_df["date"].dt.month
    all_df["weekday"] = all_df["date"].dt.dayofweek
    # 2020/01/01からの経過日数
    all_df["days_from2020"] = (all_df["date"].dt.year-2020)*365 + all_df["date"].dt.dayofyear

    # 周期を持つ関数としても追加
    # 曜日を周期関数に適用
    def m_s(x):
        return math.sin(math.radians(1.0 / 7 * x))
    def m_c(x):
        return math.cos(math.radians(1.0 / 7 * x))
    all_df['pos_circle_x_week'] = all_df["date"].dt.dayofweek.map(m_c)
    all_df['pos_circle_y_week'] = all_df["date"].dt.dayofweek.map(m_s)

    # 月を周期関数に適用
    def m_s(x):
        return math.sin(math.radians(1.0 / 12 * x))
    def m_c(x):
        return math.cos(math.radians(1.0 / 12 * x))
    all_df['pos_circle_x_month'] = all_df["date"].dt.month.map(m_c)
    all_df['pos_circle_y_month'] = all_df["date"].dt.month.map(m_s)

    to_normalize = ["inf_Lags_{}".format(i) for i in range(Lags)] + ["inf_avg{}".format(i) for i in range(7, 22, 7)] + ["inf_max{}".format(i) for i in range(7, 22, 7)] + ["Population", "month", "weekday", "days_from2020"]
    scaler = StandardScaler()
    scaler.fit( all_df[to_normalize] )

    all_df[to_normalize] = scaler.transform( all_df[to_normalize] )

    # train, public, privateに分割
    df_train = all_df.loc[range(n_train)].dropna(axis=0).reset_index(drop=True)
    df_public = all_df.loc[range(n_train, n_train+n_public)].drop("death", axis=1).reset_index(drop=True)
    df_private = all_df.loc[range(n_train+n_public, n_train+n_public+n_private)].drop("death", axis=1).reset_index(drop=True)

    # 特徴量と目的変数に分離
    train = df_train.drop(["death", "date"], axis=1)
    y_train_all = df_train.death
    train.to_feather("./output/X_train.feather")
    df_public.to_feather("./output/X_pubric.feather")
    df_private.to_feather("./output/X_private.feather")