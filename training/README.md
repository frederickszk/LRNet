# LRNet-Training

The training and evaluating codes for LRNet.

> We provide both `Tensorflow 2.x` and `PyTorch` implementation. While the `PyTorch` version is preferred. Because we customize and optimize the `Dropout` layer for input features, which is difficult to be implemented in `Tensorflow`. For further framework improvement, **we would only maintain the `PyTorch` version and we recommend you to start with it.** 





# Dataset

Before start training the model, you should prepare the datasets in the folder: `./datasets/`. Further instruction can be found on this folder's readme file.

For facilitating the training and evaluation of LRNet, we would release the processed **landmark datasets** gradually. Here is the list:

- [x] FF++ [[Github Link](https://github.com/ondyari/FaceForensics)]
  - [x] Deepfakes (DF)
    - [x] raw
    - [x] c23
    - [x] c40
  - [x] FaceSwap (FS)
    - [x] raw
    - [x] c23
    - [x] c40
  - [x] Face2Face (F2F)
    - [x] raw
    - [x] c23
    - [x] c40
  - [x] NeuralTexture (NT)
    - [x] raw
    - [x] c23
    - [x] c40
> The performance reported in the paper (CVPR'21) was obtained by training on the single `DF` dataset. While recent works tend to mix the 4 datasets for training. We have updated this training strategy on [2022/3/10]. And we update the weights trained on `FF++(raw)` (_PyTorch version only_, [2022/11/5]). You can find it at `./weights/torch/g1.pth` and `./weights/torch/g2.pth`. It can achive *99.02%* AUC.





# PyTorch

## Requirement

- PyTorch

- cudatoolkit #*(Alternative but recommended)* GPU acceleration support
- [sklearn](https://scikit-learn.org/stable/install.html#installation-instructions) # For some evaluation metric
- [matplotlib](https://matplotlib.org/stable/users/getting_started/) # For plotting the ROC curve

> The example codes have been validated on the near-latest PyTorch, the environment is:
>
> `PyTorch==1.8.0; cudatoolkit==11.0.221 `
>
> You can consult the [Official website](https://pytorch.org/) for GPU configure assistance.

## Getting started

### Notice

- Related codes files: `train.py`, `evaluate.py`, `./utils/`, `./configs/`
- Before executing the codes, please ensure that you have prepared the datasets.
- You could modify the `.yaml` files in the `./configs/` to customize the training and evaluating procedure.
  - `args_train.yaml`: Used for `train.py`
  - `args_test.yaml`: Used for `test.py`
  - `args_model.yaml`: The model's hyper-parameters
- We have provided the trained model weights in folder `./weights/torch/`. You can train and evaluate your own model by following the instructions above.
- The function `train_loop()` will return logs containing the training logs message such as the training loss of each epochs. You may use it to visualize the training procedure.


### Usage

- Basic

If you want to train a model from scratch, you can execute the following command (by default, on DF(c23)):

```shell
python train.py
```

Then you wan evaluate the model with trained weights:

```shell
python evaluate.py
```

- GPU acceleration

You can use GPU *(Recommend)* by:

```shell
python train.py -g
python evaluate.py -g
```

- Dataset selection

You can select the dataset used for training or evaluating by `-d` and `-l`. For example, executing the following command to train on NT(c23) and evaluate on DF(c23):

```shell
python train.py -d NT -l c23
python evaluate.py -d DF -l c23
```

You can also use the whole FF++ dataset (including DF, NT, F2F, FS) by:

```shell
python train.py -d FF_all
```

It will read in the real samples (Origin) 4 times to balance the amount of real and fake samples.

Noted that we carry out an inspection that not prepared dataset would result in code abortion. We would release the remaining datasets soon.

- Branch selection

You can select either branch of the LRNet to be trained or evaluated. By default, it would train/evaluate both of the branches. For instance:

```python
python train.py -b g1  # Only train the g1 branch.
python evaluate.py -b g2  # Only evaluate the g2 branch.
python train.py -b all  # The same as [python train.py]: train g1 and g2.
```

- Helper

For more details, please consult:

```shell
python train.py -h
python evaluate.py -h
```

- Jupyter Notebook

We now provide training/evaluation codes in Jupyter Notebook. 
The codes are the same as above but executed in an interactive approach.
You would open the `train.ipynb` for more details.
It's helpful in the situation that we only load the dataset once and train the model multiple times.


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

- Related codes files: `model_tf.py`, `data_utils.py`
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



# Other Infos

## Performance
> - Reported by AUC scores
> - All the results are obtained by the weights `./weights/torch/`, which are trained on `FF++(raw)`

### Sub-dataset
|  DF   |  NT   |  F2F  |  FS   |
| :---: | :---: | :---: | :---: |
| 99.79 | 97.06 | 99.35 | 99.90 |

### Different compression rate
|  raw  |  c23  |  c40  |
| :---: | :---: | :---: |
| 99.02 | 99.22 | 88.57 |

## Computational overhead
- Read in the datasets will occupy < 2GB memories and during training the codes will consume ~3.5GB. When using GPU, the GPU memory occupation will be about 1GB. ([2022/3/10]: New memories occupation reports will be updated soon.)
- When enabling GPU acceleration, training each model will complete in around 5 mins. （According to your device）
