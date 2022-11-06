# LRNet-Calibrator

The calibrator API for denoising the landmarks, which is adopted in LRNet.

# Requirements

- numpy
- [opencv-python](https://docs.opencv.org/4.5.5/da/df6/tutorial_py_table_of_contents_setup.html)



# Getting started

> Because the landmarks detection techniques are evolving, you may use better detector to preprocessing the video data and obtain the landmarks. Therefore we provide an **plug-and-use** API for calibrating the landmarks and obtaining better training data.

### Usage

The API is in package `LC.py` (Landmark Calibrator). You can call it by:

```python
from LC import calibrator
results = calibrator(video_file, landmark_sequence)
```

The `video_file` is your video file name end with ".mp4", etc. The `landmark_sequence` is a sequence contains the landmark positions in each video frame with the shape of ![](http://latex.codecogs.com/gif.latex?N\times68\times2). And this API return the calibrated and aligned landmark sequence `result` with the shape of ![](http://latex.codecogs.com/gif.latex?N\times136). Note that the `result` is also normalized into [-1, 1], which can be used for training and testing the model.



### Example

For easier to understand the process, we provide an example in `./example/` folder. This video comes from DF(c23), and the corresponding landmark results are detected by [[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)] toolkit. We can achieve a successful API call by:

```python
from LC import calibrator, readin_csv
video_name = "./example/000_003"
landmark_sequence = readin_csv(video_name)
results = calibrator(video_name + '.mp4', landmark_sequence)
np.savetxt(video_name+".txt", results, fmt='%1.5f')
```

Then you would see a new `000_003.txt` in folder `./example/`

You can also uncomment the example code in `LC.py` (L288~291) and directly run it to get the same result.



### Utils
We now provide utils for processing a whole dataset (in `utils.py`). The requirements are:
- tqdm
- numpy

The utils are designed for the workflow that the facial 2D landmarks are extracted by [[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)] toolkit.
In this pipeline, each landmark sequence (corresponding to a video) is recorded in a csv file. 
We can carry out the calibration on the dataset as follows:

```python
from uitls import calibrate_batch

video_path = [The directory of the video datasets]
csv_path = [The directory of the extracted landmarks (csv files)]
output_path = [The directory to save the calibrated landmarks (txt files)]

calibrate_batch(video_path, csv_path, output_path)
```
