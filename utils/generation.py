import numpy as np


def generate(G, source):
    """
    画像データを生成
    # 引数
        G : Keras model, 生成器
        source : List or Image
    # 戻り値
        images : Numpy array, 画像データ
    """
    input_dim = G.input_shape[1]
    images = G.predict(source)
    images = images * 255
    return images