import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# KNNレコメンド
# データcsvファイル
# パターン:協調フィルタリング 〇〇が好きな人はこれも好き　他人のスコアリングを元に算出
# 分類api本体

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["POST"])
# @app.route("/", methods=["GET"])
def pridect():

    reccomend_predict = request.json['reccomend']
    print(reccomend_predict)

    anime_test = pd.read_csv('test_anime2.csv')
    anime_pivot2 = anime_test.pivot(
        index='name', columns='user_id', values='rating').fillna(0)
    anime_pivot_sparse2 = csr_matrix(anime_pivot2.values)

    knn = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')

    # 前処理したデータセットでモデルを訓練
    model_knn2 = knn.fit(anime_pivot_sparse2)

    #Anime = '京都'
    Anime = reccomend_predict

    distance, indice = model_knn2.kneighbors(
        anime_pivot2.iloc[anime_pivot2.index == Anime].values.reshape(1, -1), n_neighbors=5)
    dict_dist = {}

    for i in range(0, len(distance.flatten())):
        if i == 0:
            print('Recommendations if you like  {0}:\n'.format(
                anime_pivot2[anime_pivot2.index == Anime].index[0]))
        else:
            print('{0}: {1} with distance: {2}'.format(
                i, anime_pivot2.index[indice.flatten()[i]], distance.flatten()[i]))
            dict_dist[distance.flatten()[i]
                      ] = anime_pivot2.index[indice.flatten()[i]]

    return jsonify(dict_dist)


if __name__ == "__main__":
    app.run()
