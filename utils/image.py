import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def to_dirname(name):
    """
    ディレクトリ名の"/"有無の違いを吸収する
    # 引数
        name : String, ディレクトリ名
    # 戻り値
        name : String, 変更後
    """
    if name[-1:] == '/':
        return name
    else:
        return name + '/'


def check_dir(name):
    """
    ディレクトリの存在を確認して、存在しなければ作成する
    # 引数
        name : String, ディレクトリ名
    """
    if os.path.isdir(name) == False:
        os.makedirs(name)


def save_images(images, name, ext='.jpg'):
    """
    画像群を任意の場所に保存する
    # 引数
        images : Numpy array, 画像データ
        name : String, 保存場所
        ext : String, 拡張子
    """
    check_dir(name)
    # PILで保存できるように型変換
    images = images.astype(np.uint8)
    for i in range(len(images)):
        image = Image.fromarray(images[i])
        # "*/result[0-9]*.jpg" の形で保存
        image.save(name+'/result'+str(i)+ext)


def load_images(name, size, ext='.jpg'):
    """
    画像群を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
        ext : String, 拡張子
    # 戻り値
        images : Numpy array, 画像データ
    """
    images = []
    #images = np.empty((0, size[0], size[1], 3))
    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:
            # 拡張子が違うなら処理しない
            continue
        image = Image.open(name+file)
        if image.mode != "RGB":
            # 3ch 画像でなければ変換する
            image.convert("RGB")
        image = image.resize(size)
        image = np.array(image)
        images.append(image)
        #images = np.concatenate((images, [image]))
    images = np.array(images)
    # 256階調のデータを0-1の範囲に正規化する
    images = images / 255
    return images