import numpy as np
from argparse import ArgumentParser
from keras.optimizers import SGD, Adam
# 自作関数群
from utils.file import save_model
from utils.image import save_images, load_images, to_dirname
from utils.generation import generate
from utils.training import train, train_with_images
# 使用ネットワーク
from networks.leakyrelu import build_generator, build_upsampler, build_discriminator, build_GAN


def get_args():
    description = 'Build DCGAN models and train'
    parser = ArgumentParser(description=description)
    parser.add_argument('-d', '--dim', type=int, default=100, help='generator input dimension')
    parser.add_argument('-z', '--size', type=int, nargs=2, default=[64, 64], help='image size during training')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=3000, help='number of epochs')
    parser.add_argument('-s', '--save', type=int, default=100, help='snapshot taking interval')
    parser.add_argument('-i', '--input', type=str, default='images', help='data sets path')
    parser.add_argument('-o', '--output', type=str, default='gen', help='output directory path')
    return parser.parse_args()


def main():
    args = get_args()
    # パラメータ設定
    input_dim = args.dim # 入力ベクトルサイズ
    image_size2x = args.size # 画像サイズStage2
    image_size = (image_size2x[0]//2, image_size2x[1]//2) # 画像サイズStage1
    batch = args.batch # 勾配更新までの回数
    epochs = args.epoch # データを周回する回数
    save_freq = args.save # スナップショットのタイミング
    input_dirname = to_dirname(args.input) # 読み込み先ディレクトリ
    output_dirname = to_dirname(args.output) # 出力先ディレクトリ
    # Stage 1
    G1 = build_generator(input_dim=input_dim, output_size=image_size)
    D1 = build_discriminator(input_size=image_size)
    optimizer = Adam(lr=1e-5, beta_1=0.1)
    D1.compile(loss='binary_crossentropy', optimizer=optimizer)
    GAN1 = build_GAN(G1, D1)
    optimizer = Adam(lr=1e-4, beta_1=0.5)
    GAN1.compile(loss='binary_crossentropy', optimizer=optimizer)
    # Stage 2
    G2 = build_upsampler(input_size=image_size)
    D2 = build_discriminator(input_size=image_size2x)
    optimizer = Adam(lr=1e-5, beta_1=0.1)
    D2.compile(loss='binary_crossentropy', optimizer=optimizer)
    GAN2 = build_GAN(G2, D2)
    optimizer = Adam(lr=1e-4, beta_1=0.5)
    GAN2.compile(loss='binary_crossentropy', optimizer=optimizer)
    # モデルを保存
    save_model(G1, 'G1_model.json')
    save_model(D1, 'D1_model.json')
    save_model(G2, 'G2_model.json')
    save_model(D2, 'D2_model.json')
    # データセットを読み込み
    images = load_images(name=input_dirname, size=image_size)
    images2x = load_images(name=input_dirname, size=image_size2x)
    # 学習開始
    for epoch in range(epochs):
        # Stage 1
        print('Epoch: '+str(epoch+1)+'/'+str(epochs)+' - Stage: 1')
        train(G1, D1, GAN1, sets=images, batch=batch)
        # Stage 2
        print('Epoch: '+str(epoch+1)+'/'+str(epochs)+' - Stage: 2')
        train_with_images(G1, G2, D2, GAN2, sets=images2x, batch=batch)
        if (epoch + 1) % save_freq == 0:
            # 一定間隔でスナップショットを撮る
            noise = np.random.uniform(0, 1, (batch, input_dim))
            results1 = generate(G1, source=noise)
            save_images(results1, name=output_dirname+'stage1/'+str(epoch+1))
            results2 = generate(G2, source=results1/255)
            save_images(results2, name=output_dirname+'stage2/'+str(epoch+1))
            G1.save_weights('G1_weights.hdf5')
            D1.save_weights('D1_weights.hdf5')
            G2.save_weights('G2_weights.hdf5')
            D2.save_weights('D2_weights.hdf5')


if __name__ == '__main__':
    main()