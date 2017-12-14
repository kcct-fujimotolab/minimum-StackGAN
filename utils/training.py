import math
import numpy as np


def display(stage, step, steps, D_loss, GAN_loss):
    """
    学習の進行状況を表示する
    # 引数
        step : Integer, 現在のステップ
        steps : Integer, ステップ数
        D_loss : Float, 損失値
        GAN_loss : Float, 損失値
    """
    print('\rStep: '+str(step+1)+'/'+str(steps), end='')
    print(' - D loss: '+str(D_loss)+' - GAN loss: '+str(GAN_loss), end='')


def train(G, D, GAN, sets, batch):
    """
    ネットワークの学習を行う
    # 引数
        G : Keras model, 生成器
        D : Keras model, 判別器
        GAN : Keras model, GAN
        sets : 学習用データセット
        batch : Integer, バッチサイズ
    # 戻り値
        loss : Tuple, 損失値
    """
    # データセットをシャッフル
    np.random.shuffle(sets)
    input_dim = G.input_shape[1]
    # バッチサイズからループ回数を求める
    steps = math.ceil(len(sets) / batch)
    for step in range(steps):
        # データセットからバッチサイズだけ抽出
        real = sets[step*batch:(step+1)*batch]
        samples = len(real)
        # 本物画像の学習
        answer = np.ones(samples)
        D_loss = D.train_on_batch(x=real, y=answer)
        # 偽物画像の学習
        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        generated = G.predict(noise)
        answer = np.zeros(samples)
        D_loss = D.train_on_batch(x=generated, y=answer)
        # 生成器の学習
        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        answer = np.ones(samples)
        GAN_loss = GAN.train_on_batch(x=noise, y=answer)
        # 進捗表示
        display(step, steps, D_loss, GAN_loss)
    print()
    return (D_loss, GAN_loss)


def train_with_images(G_before, G, D, GAN, sets, batch):
    """
    ネットワークの学習を行う
    # 引数
        G_before : Keras model, 前段の生成器
        G : Keras model, 生成器
        D : Keras model, 判別器
        GAN : Keras model, GAN
        sets : 学習用データセット
        batch : Integer, バッチサイズ
    # 戻り値
        loss : Tuple, 損失値
    """
    # データセットをシャッフル
    np.random.shuffle(sets)
    input_dim = G_before.input_shape[1]
    # バッチサイズからループ回数を求める
    steps = math.ceil(len(sets) / batch)
    for step in range(steps):
        # データセットからバッチサイズだけ抽出
        real = sets[step*batch:(step+1)*batch]
        samples = len(real)
        # 本物画像の学習
        answer = np.ones(samples)
        D_loss = D.train_on_batch(x=real, y=answer)
        # 偽物画像の学習
        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out = G_before.predict(noise)
        generated = G.predict(G_out)
        answer = np.zeros(samples)
        D_loss = D.train_on_batch(x=generated, y=answer)
        # 生成器の学習
        noise = np.random.uniform(0, 1, size=(samples, input_dim))
        G_out = G_before.predict(noise)
        answer = np.ones(samples)
        GAN_loss = GAN.train_on_batch(x=G_out, y=answer)
        # 進捗表示
        display(step, steps, D_loss, GAN_loss)
    print()
    return (D_loss, GAN_loss)