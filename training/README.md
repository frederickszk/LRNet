# LRNet-Training

The training and evaluating codes for LRNet.

> We provide both `Tensorflow 2.x` and `PyTorch` implementation. While the `PyTorch` version is preferred. Because we customize and optimize the `Dropout` layer for input features, which is difficult to be implemented in `Tensorflow`. For further framework improvement, **we would only maintain the `PyTorch` version and we recommend you to start with it.** 



# Dataset

Before start training the model, you should prepare the datasets in the folder: `./datasets/`. Further instruction can be found on this folder's readme file.

For facilitating the training and evaluation of LRNet, we would release the processed **landmark datasets** gradually. Here is the list:

- [ ] FF++ [[Github Link](https://github.com/ondyari/FaceForensics)]
  - [ ] Deepfakes (DF)
    - [x] c23
  - [ ] FaceSwap (FS)
    - [x] c23
  - [ ] Face2Face (F2F)
    - [x] c23
  - [ ] NeuralTexture (NT)
    - [x] c23

> This example code are training on single DF(c23). While recent works tend to mix the 4 datasets for training. We would update this in the future.





# TensorFlow 2.x

## Requirement

- TensorFlow > 2.0

*(Recommend)* GPU acceleration support:

- cudatoolkit
- cudnn

> The example codes have been validated on the latest TF2, the environment is:
>
> `TensorFlow==2.6.2; cudatoolkit==11.3.1; cudnn==8.2.1 `
>
> You can simply configure it by:
>
> ```bash
> pip install tensorflow
> conda install cudnn
> ```



## Getting started

### Notice

- Before executing the codes, please ensure that you have prepared the datasets, and check the **initialization area (L73---L82)** in `model_tf.py`.
- Because LRNet have 2 branches `g1` and `g2`, which are trained and tested separately, you can select them freely by commenting or uncommenting them in code blocks between `# ----For g1/2----#` and `# ----g1/2 end----#`. **(L154, L168, L188, L194)**.
- We have provided the trained model weights for exhibition in folder `./weights/tf/`. Back-up them if needed before training or modify the codes **(L157, L171, L190, L196)** to save the weights to your own files. Otherwise if you directly start the training, the example weights will be overwritten.



### Usage

If you want to train a model from scratch, you can execute the following command:

```shell
python model_tf.py -t
```

Then you wan evaluate the model with trained weights:

```shell
python model_tf.py -e
```

Also you can use GPU *(Recommend)* by:

```shell
python model_tf.py -t -g
```

You can carry out the whole training and evaluation procedures in one command:

```shell
python model_tf.py -t -e -g
```

For details, please consult:

```shell
python model_tf.py -h
```





# PyTorch

## Requirement

- PyTorch

*(Recommend)* GPU acceleration support:

- cudatoolkit

> The example codes have been validated on the near-latest PyTorch, the environment is:
>
> `PyTorch==1.8.0; cudatoolkit==11.0.221 `
>
> You can consult the [Official website](https://pytorch.org/) for GPU configure assistance.



## Getting started

### Notice

- Before executing the codes, please ensure that you have prepared the datasets, and check the **initialization area (L202---L208)** in `model_torch.py`.
- Because LRNet have 2 branches `g1` and `g2`, which are trained and tested separately, you can select them freely by commenting or uncommenting them in code blocks between `# ----For g1/2----#` and `# ----g1/2 end----#`. **(L256, L265, L283, L289)**.
- We have provided the trained model weights for exhibition in folder `./weights/torch/`. Back-up them if needed before training or modify the codes **(L261, L270, L284, L290)** to save the weights to your own files. Otherwise if you directly start the training, the example weights will be overwritten.
- The function `train()` (L147) will return logs containing the training logs message such as the training loss of each epochs. You may use it to visualize the training procedure.



### Usage

If you want to train a model from scratch, you can execute the following command:

```shell
python model_torch.py -t
```

Then you wan evaluate the model with trained weights:

```shell
python model_torch.py -e
```

Also you can use GPU *(Recommend)* by:

```shell
python model_torch.py -t -g
```

You can carry out the whole training and evaluation procedures in one command:

```shell
python model_torch.py -t -e -g
```

For details, please consult:

```shell
python model_torch.py -h
```



# Other Infos

- Read in the datasets will occupy < 2GB memories and during training the codes will consume ~3.5GB. When using GPU, the GPU memory occupation will be about 1GB.
- When enabling GPU acceleration, training each model will complete in 3~4 mins. （According to your device）