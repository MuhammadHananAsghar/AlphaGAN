# AlphaGAN
A Stable GAN for 3-Channels Images

## Description
AlphaGAN usually supports 32px, 64px, 128px images. Usually training to the 20-25 epochs we got results.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Latest Tensorflow Version.
```bash
pip install tensorflow
```

## Important Parameters
```
SIZE = 32  # 32, 64, 128
SEED = 100  # 100, 200, 300
IMAGE_SHAPE = (SIZE, SIZE, 3)
IMAGES = "/"
IMAGES_PATH = "/"
OUTPUT_PATH = "/"
MODEL_PATH = '/'
BATCH_SIZE = 96
```
## Images
Some of the images produced by AlphaGAN.

## Animes
![Animes](alphagan/images/anime.png)
## Faces
![Faces](alphagan/images/faces.png)
![Faces](alphagan/images/generated_plot.png)


```
Implemented By Muhammad Hanan Asghar
```
