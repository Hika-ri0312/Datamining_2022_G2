import pandas as pd

def load_imgs():
    """ 画像データを2つのpickleファイルから読み込み、必要なデータのみnumpy配列に保存
    return:
        imgs1 (numpy.ndarray): 1つ目のファイルの画像データ。2次元配列になっている。
        imgs2 (numpy.ndarray): 2つ目のファイルの画像データ。2次元配列になっている。

    Examples:
        >>> a, b = load_imgs()
        >>> type(a)
        <class 'numpy.ndarray'>
        >>> len(a.shape)
        2
     """
    path1 = "./dataset_code/dataset/ukiyoe_grayImg.pkl"
    path2 = "./dataset_code/dataset/met_data.pkl"
    df1_temp = pd.read_pickle(path1)
    df2_temp = pd.read_pickle(path2)

    imgs1=df1_temp["images"]

    imgs2=df2_temp["images"]
    
    return imgs1, imgs2


def get_plickle():
    """ 画像データ1には Ukiyoe のラベル、画像データ2には Western のラベルをつける
    val:
        imgs1 (numpy.ndarray): 2次元配列.
        imgs2 (numpy.ndarray): 2次元配列.
        df1 (pandas.DataFrame): imgs1 のに Ukiyoe ラベルをつけたもの
        df2 (pandas.DataFrame): imgs2 のに Western ラベルをつけたもの
        df (pandas.DataFrame): df1 と df2 を合わせたもの
     """
    imgs1, imgs2=load_imgs()

    df1=pd.DataFrame({"images": iter(imgs1)})
    df1.insert(len(df1.columns) ,"Class", "Ukiyoe")

    
    df2=pd.DataFrame({"images": iter(imgs2)})
    df2.insert(len(df2.columns) ,"Class", "Western")

    df = pd.concat([df1, df2], ignore_index=True)

    df.to_pickle("./dataset_code/true_dataset/ukiyoe_and_western.pkl")
    print(f"imgs: \n{df}")
    
    
    
if __name__=="__main__":
    get_plickle()