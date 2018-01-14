# StackGAN

Implementation of "なんちゃって" StackGAN model using Keras.

## Discription

"なんちゃって" is a phrase that means that it is not genuine.

## Requirements

- Python 3.0 or newer
- Keras 2.0 or newer (Tensorflow backend)
- Pillow
- numpy
- tqdm
- h5py

## Getting started

1. Clone this repository:
```sh
git clone https://github.com/kcct-fujimotolab/StackGAN.git
cd DCGAN/
```

2. Make a directory for data sets:
```sh
mkdir images
```

3. Collect images (more than 2000-3000 works better):
```sh
ls images/
data0000.jpg   data0001.jpg   ...   data9999.jpg
```

4. Start training by specifying image size, number of epochs, data set directory, etc.:
```sh
python train.py --input images/ --size 128 128 --epoch 5000
```

## Options

`--help` `-h`: show information

### train.py

`--input` `-i`: data sets path (default `-i images/`)  
`--size` `-z`: image size during training, **2 values required**, **must be multiples of 8** (default `-z 128 128`)  
`--epoch` `-e`: number of epochs (default `-e 3000`)  
`--batch` `-b`: batch size (default `-b 64`)  
`--dim` `-d`: input dimension of generator (default `-d 100`)  
`--output` `-o`: output directory path (default `-o gen/`)  
`--save` `-s`: taking snapshot interval (default `-s 100`)

## Results

We extracted 4096 images from the face data provided by [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/), and trained with Keras model.

### 1000 epochs
![1000](https://i.imgur.com/LlPjdKG.jpg)

### 2000 epochs
![2000](https://i.imgur.com/bI7mQF2.jpg)

### 4000 epochs
![4000](https://i.imgur.com/MrqoC6a.jpg)

### 6000 epochs
![6000](https://i.imgur.com/R8XHxl5.jpg)

## Author

[Fujimoto Lab](http://www.kobe-kosen.ac.jp/~fujimoto/) at [Kobe City College of Technology](http://www.kobe-kosen.ac.jp)  
Undergraduate Student of Electronics Department  
[@yoidea](https://twitter.com/yoidea)