import pandas as pd
import numpy as np

def read_pickle():
    """ ukiyoe_and_western.pklを読み込む
    return:
        X (list): 特徴ベクトル. 2重リスト.
        y (list): 教師データ. 1重リスト.

    examples:
    >>> a, b = read_pickle()
    >>> type(a)
    <class 'list'>
    >>> len(a[0])
    4,096
     """
    path = "./dataset_code/true_dataset/ukiyoe_and_western.pkl"
    df=pd.read_pickle(path)
    new_df=df.dropna(how='any')

    X=[]
    for data in list(new_df["images"]):
        X.append(data.tolist())

    y=new_df["Class"].values

    return X, y



def load_dataset(n=0, m=-1):
    """ 学習データセットの特徴ベクトル X と、教師データ y を取得する
    args:
        n (int): 学習データを取得するときに、データのn番目から取得できる
        m (int): 学習データを取得するときに、データのm番目まで取得できる

    return:
        x (numpy.ndarray): 特徴ベクトル. 2重配列. 学習に使う画像データ.
        y (numpy.ndarray): 教師データ. 1重配列. Ukiyoe か Western のラベル.
    
    Examples:
        >>> a, b = load_dataset()
     """
    x, y=read_pickle()
    
    #x=np.array(x)
    #y=np.array(y)
    x=np.array(x[n:m])
    y=np.array(y[n:m])

    return x, y


if __name__ == "__main__":
    x, y =load_dataset()
    print(x.shape)
    print(y.shape)
    print(x[0:5])
    print(y[0:5])

