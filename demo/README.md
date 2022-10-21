# LRNet-Demo
The demo of *LRNet*. It reproduces the whole procedure from a input video sample to its corresponding predicted label. 

# Folders & Codes

`./input`

The input sample videos. Here we provide a real and a fake video.

`./landmarks`

The extracted landmarks. Each video is corresponding to a .txt file.

> This folder contains the old version landmarks that only applicable to the old version model in `classify.py `. It would be replaced when the new version model is updated.

`./landmarks_new`

Temporary folder for new version facial landmarks.

`./model_weights`

The two-stream RNN's weight. We provide two sets of weights here, you can check Log 2021/5/18 for more details (in `LRNet/README.md`).

`./visualize`

The visualize results of landmrks extraction. The original landmarks detection (without calibration) is shown in "_origin.avi", and the calibrated one is shown in ".track.avi".

`./utils/`

Several utility functions. The structure of the files and their respective usages are as follows:

```css
|-- utils
    |-- FaceDetector 
        /*The RetinaFace face detector*/
    |-- landmark_utils.py 
        /*Supports the landmark extraction and calibration*/
    |-- shared.py 
        /*Several global variables*/
    |-- deprecated.py 
        /*Deprecated codes, for reference only*/
```

---

`extract_landmarks.py`

Perform the landmarks extraction and calibration.

`classify.py`

Perform the classification on extracted landmarks. Including reading and embedding the data, building the two-stream RNN and inference the samples.


# Requirements

## For landmarks extraction

- numpy
- [opencv-python](https://docs.opencv.org/4.5.5/da/df6/tutorial_py_table_of_contents_setup.html)
- tqdm
- [face-alignment](https://github.com/1adrianb/face-alignment)
  - The facial landmark detector. Also noted as `FAN`.
  - It includes a high-speed face detector `blazeface`, which we use by default.
- [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) **(OPTIONAL)**
  - We have intergrated this `retinaface` face detector, which provides the option to trade more computing time for better face detection results.
  - Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.
  - You could swith the backbones between `mobilenet0.25` and `resnet50`. We use the `mobilenet0.25` by default, and no additional configuration is required.
  - If you want to set the `resnet50` as backbone, you need to complete the following configuration steps:
    - Download the model weghts `Resnet50_Final.pth` from [google cloud](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) or [baidu cloud](https://pan.baidu.com/share/init?surl=2h97Fy1RYuqMMIV-RpzdPg) (passwd: fstq).
    - Place it under the `./utils/FaceDetector/weights`.
    - Modified the `trained_model` and `network` in the configure file `./utils/FaceDetector/config/args_face_detector.yaml`.


## For classification

- numpy
- tqdm
- TensorFlow >2.0
    - CPU version is sufficient



# Useage

## Landmarks extraction

To extract the landmarks from input videos, please ensure that all the samples are placed in `./input` folder and run

```python
python extract_landmarks.py
```

You can also customize it by:
- Specific your own input videos path by `-i` or output landmarks path by `-o`
- Use `-v` to enable the visualization, and specific the visualization video output path by `--visualize_path`
- Specific the log file's name by `-l`
- Select the face detector by `--fd`


For detailed, please consult

```python
python extract_landmarks.py -h
```

> 1. **(IMPORTANT)** There is a major change in the current version: We no longer align the landmarks but normalize them into [-1, 1], which shows better performance. Therefore they cannot be directly input into `classify.py` and we then put them in another folder. Currently, we still preserve the data and models of the old version, and we would update them in these days (also the new version datasets).
> 2. We add a log file that records the average prediction score of the landmarks in a video. It is useful to exclude some poor quality videos.
> 3. We add a breakpoint mechanism that would be helpful in deployment environments. It would skip the videos that have landmarks result in the output path.

## Classification

Before classification, make sure that the landmarks have been extracted and put in `./landmarks` folder.

Execute the following command:

```python
python classify.py
```

and you can see the prediction results (predicted label and score of the corresponding sample. Noticed the score that "0=real" and "1=fake".
