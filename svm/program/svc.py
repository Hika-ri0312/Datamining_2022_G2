import dataset
from sklearn import metrics 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix

def SVC(x,y):
    """ 外部ライブラリのsklearnより学習モデルSVCによって、学習を行う。

    args:
        x (numpy.ndarray): 特徴ベクトル. 2重配列になっている.
        y (numpy.ndarray): 教師データ. 1重配列.

    return:
        None

    val:
        model (sklearn.svm.SVC): SVCモデル. 

    Note:
        調整対象のハイパーパラメータ
        C       : 正則化パラメータ。値が小さいほど誤りを許容する。

        精度の評価方法
        preicision  :
            適合率。モデルが真と予測した数を分母、その中で実際に正解した数を分子にした値。
            preicision = TP/(TP+TP)

        recall      : 
            再現率。正解データ中の真の数を分母、その中でモデルが正解した数を分子にした値。
            recall = TP/(TP/FN)

        f1-score    : 
            F値。precisionとrecallの調和平均。
            f1-score = 2*precision*recall/(precision+recall)
        
        support     : 正解データに含まれている個数。
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30 , random_state=0)
    
    parameters = {
        'C': [0.01, 0.1, 1, 10, 100]
    }
    
    # 学習と予測
    
    #モデルインスタンス
    model = SVC(max_iter = 100, random_state=0) #グリッドサーチのみ高速で行うために反復回数を100に制限
    
    #ハイパーパラメーターチューニング（グリッドサーチのコンストラクタにモデルと辞書パラメータを指定)
    gridsearch = GridSearchCV(estimator = model,         #モデル
                            param_grid = parameters,   #チューニングするハイパーパラメータ
                            scoring = "accuracy")      #スコアリング

    
    #演算実行
    gridsearch.fit(x_train, y_train)
    
    # グリッドサーチの結果から得られた最適なパラメータ候補を確認
    print('----------------------------------------------------')
    print('Best params: {}'.format(gridsearch.best_params_)) 
    print('Best Score: {}'.format(gridsearch.best_score_))
    print('----------------------------------------------------')
    
     # 最適なハイパーパラメータの組み合わせを用いてモデル再構築
    model = SVC(C = gridsearch.best_params_['C'],
                random_state = 0, # 乱数シード
                )
    
    print(model)
    # モデル学習
    model.fit(x_train,y_train)

    
    predict = model.predict(x_test)
    

    # 精度を確認
    ac_score = metrics.accuracy_score(y_test, predict)#予測結果が正解ラベルと同じである割合を算出
    cl_report = metrics.classification_report(y_test, predict) #ラベル毎の精度を求める
    print("正解率=", ac_score)
    print("レポート=\n", cl_report)
    
    print(len(x_train),len(y_train))
    visualizer = ConfusionMatrix(model)

    visualizer.fit(x_train, y_train)
    visualizer.score(x_test, y_test)
    visualizer.poof();

def main():
    x,y = dataset.load_dataset()
    SVC(x,y)
    
    
    
if __name__ == '__main__':
    main()