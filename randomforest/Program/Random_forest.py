# -*- coding: utf-8 -*-
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV #データ分割用,グリッドサーチ
from sklearn.ensemble import RandomForestClassifier #ランダムフォレスト

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import doctest

import dataset

# 学習用とテスト用データに分ける
def random_forest(x,y):
    """ランダムフォレスト
    Args:
        x (numpy.ndarray): 特徴ベクトル. 2重配列
        y (numpy.ndarray): 教師データ. 1重配列
        
    val:
        model   (sklearn.ensemble._forest.RandomForestClassifier): ランダムフォレスト
    
    Note:
    ---------
    調整対象のハイパーパラメータ
    n_estimators:用意する決定木の数
            （10,20,30,50,100,300）
    
    max_depth:決定木のノード深さの制限値
            (3,10,20,30,40,50,None)
    
    精度の評価方法
    precision  : 適合率
        分母はモデルが真であると予測した数、分子はその中で実際に正解した数を表す。
        precision = TP / (TP + FP)
        
    recall     : 再現率 
        分母は正解データ内の真の数、分子はその中でモデルが正解した数を表す。
        recall = TP / (TP + FN)
        
    f1-score   : F値
        precisionとrecallの調和平均
        f1-score = ( 2 * precision * recall ) / (precision + recall)

    support    : 正解データに含まれている個数
    
    """
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30 , random_state=0)
    
    parameters = {
        'n_estimators':[10,20,30,50,100,300],           #用意する決定木
        'max_depth':(3,10,20,30,40,50,None),              #決定木のノード深さの制限
        
    }
    
    # 学習と予測
    
    #モデルインスタンス
    model = RandomForestClassifier()
    
    #ハイパーパラメーターチューニング（グリッドサーチのコンストラクタにモデルと辞書パラメータを指定)
    gridsearch = GridSearchCV(estimator = model,         #モデル
                            param_grid = parameters,   #チューニングするハイパーパラメータ
                            scoring = "accuracy")      #スコアリング

    
    #演算実行
    gridsearch.fit(x_train, y_train)
    
    # グリッドサーチの結果から得られた最適なパラメータ候補を確認
    print('グリットサーチの結果 ====================================================')
    print('Best params: {}'.format(gridsearch.best_params_)) 
    print('Best Score: {}'.format(gridsearch.best_score_))
    
    
    # 最適なハイパーパラメータの組み合わせを用いてモデル再構築
    model = RandomForestClassifier(n_estimators = gridsearch.best_params_['n_estimators'], # 用意する決定木モデルの数
                               max_features = 'sqrt', # ランダムに指定する特徴量の数
                               max_depth    = gridsearch.best_params_['max_depth'],    # 決定木のノード深さの制限値
                               criterion='gini',                                       # 不純度評価指標の種類(ジニ係数gini）
                               min_samples_leaf = 1,                                   # 1ノードの深さの最小値
                               random_state = 0,                                       # 乱数シード
                              )
    
    print(f'学習モデル ==================================================== \n{model}')
    # モデル学習
    model.fit(x_train,y_train)

    
    predict = model.predict(x_test)
    
    #混同行列描画
    y_pred = model.predict(x_test) # テストデータを用いて予測値を算出
    m = confusion_matrix(y_test,y_pred)
    print(f'Confusion matrix ==================================================== \n{m}')
    

    # 精度を確認
    ac_score = metrics.accuracy_score(y_test, predict)#予測結果が正解ラベルと同じである割合を算出
    cl_report = metrics.classification_report(y_test, predict) #ラベル毎の精度を求める
    print('精度の詳細 ====================================================')
    print("正解率=", ac_score)
    print("レポート=\n", cl_report)
    
    print(len(x_train),len(y_train))
    
    
    sns.heatmap(m,square = True, cbar = True, annot=True, cmap='Blues')
    plt.savefig('image_sample.jpg')
    plt.show()
    

    
    
    
def main():
    '''
    >>> x,y = dataset.load_dataset()
    >>> type(x)
    <class 'numpy.ndarray'>
    >>> type(y)
    <class 'numpy.ndarray'>
    '''
    x,y = dataset.load_dataset()
    
    random_forest(x,y)
    
    
    
if __name__ == '__main__':
    main()
    doctest.testmod()
    