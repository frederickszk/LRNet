# LRNet-Training-Datasets

For placing the dataset files.

> [22-11-05] Update
> - We now adopt the dataset-split according to the original [FF++](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits): `train(720):val(140):test(140)`. During training we use `val` to determine the checkpoints, and the saved weights are evaluate by the `test` dataset, which avoids the identity leak.
> - The current dataset uses a [**new normalization method**](https://github.com/frederickszk/LRNet/blob/d090c80c9e14b28c2e293c4ba5f3e1e2b79bf5de/demo/utils/landmark_utils.py#L321), which discards the previous landmark-alignment, so developers using the old dataset would need to **redownload** the dataset.
> 
> [24-01-07] "Stable" dataset
> - We update the dataset where the landmarks become more 'stable', leading to a better performance. The details would be updated soon.
> - For convenience, we do not split the dataset into 3 compression levels. Instead, we provide the download link for the whole dataset.

# File structure

Each folder in this root directory represents a specific datasets. The `Origin` represents the real face dataset in `FF++`. And for the other abbreviations of dataset you can check the readme file in the upper directory.

The structure of `Origin`, `DF`, `FS`, `F2F`, `NT` should be like (take `DF` as example)

```css
|-- DF
    |-- raw
        |-- train
            < 001_870.txt, ..., 999_960.txt >
        |-- val
            < 004_982.txt, ..., 992_980.txt >
        |-- test
            < 000_003.txt, ..., 995_233.txt >
    |-- c23
        |-- ...
    |-- c40
```

Therefore, you can download the dataset (link listed below) and copy the `raw/c23/c40` folder into corresponding folder, or just overwrite the folders.

For your own landmark dataset, you could organize them in similar structure and adjust the training codes accordingly. 



# Download links
> For FF++, we recommend using `raw`-level for training, as it has the best quality.

## Google Drive

### FF++

- [Download Link](https://drive.google.com/file/d/1_w3dcK18wqPkir8RmT4VB8bmG_6DNl1t/view?usp=drive_link)


## Baidu Net-disk

### FF++

- [Download Link](https://pan.baidu.com/s/17nALjof9hbC2u-rh3b9zew?pwd=s5sx) (提取码：s5sx)

