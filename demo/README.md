# LRNet-Demo
The demo of *LRNet*. It reproduces the whole procedure from a input video sample to its corresponding predicted label. 
> **[22-11-05]** Update the landmark extraction in `extract_landmark.py`.
> - Add a log file that records the average prediction score of the landmarks in a video. It is useful to exclude some poor quality videos.
> - Add a breakpoint mechanism that would be helpful in deployment environments. It would skip the videos that have landmarks result in the output path.
> 
> **[24-01-08]** Important Update.
> - Update the model to PyTorch version. You could download the new weights.
> - Optimize the stability of landmark extraction in `utils/landmark_utils.py`. You could check Log 2024/1/8 for more details (in `LRNet/README.md`)

# Folders & Codes
`./configs`

The configuration files for LRNet model. 

`./input`

The input sample videos. Here we provide two sets of real and fake video.

One is from the FF++ dataset (marked as `ff`), the other is from the Deeperforensics-1.0 (`dfo`).

`./landmarks`

The extracted landmarks. Each video is corresponding to a .txt file.


`./model_weights`

The two-stream RNN's weight. We provide the weights trained on `dfo` because it has better generalization ability.

`./visualize`

The visualize results of landmrks extraction. The original landmarks detection (without calibration) is shown in "_origin.avi", and the calibrated one is shown in ".track.avi".
To save the space, we only provide 1 example.

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
- [Pytorch](https://pytorch.org/get-started/locally/)
  - Verified on the latest version (Pytorch 2.1.2 with CUDA 12.1)
  - CPU version is also supported


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


## Classification

Before classification, make sure that the landmarks have been extracted and put in `./landmarks` folder.

Execute the following command:

```python
python classify.py
```

and you can see the prediction results (predicted label and score of the corresponding sample. Noticed the score that "0=real" and "1=fake".

Besides, you could modify the configs in `./configs/args_inference.yaml` to assign your onw model weights, input data, etc.