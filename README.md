# Few-shot-segmentation
A deep learning framework for brain tumor segmentation
We use the BraTS 2020 dataset to verify the performance of our proposed few-shot learning architecture. Our method only utilizes two adjacent images, instead of the target image, as the input data of deep neural network to predict the brain tumor area in the target image. Avoiding noise interference in the target image, this method makes full use of the spatial context feature between adjacent slices in order to obtain accurate zero-shot segmentation results.
The flowchart of our architecture can be seen as follows:

images/1.jpg

The flowchart of our novel method. (a) This is the first architecture that does not use target images for medical image segmentation as far as we know. (b) It is extremely robust to the noise in the target image. (c) Our method has lower hardware requirements, and it is easy to train.

## Prerequisites
The following packages are required for executing the main code file:

NumPy http://www.numpy.org/

Pandas https://pandas.pydata.org/

Scikit-learn http://scikit-learn.org/stable/install.html

Tensorflow https://tensorflow.google.cn/

SimpleITK https://simpleitk.org/

Nibabel https://pypi.org/project/nibabel/

h5py https://pypi.org/project/h5py/
  
## Datasets
We train the network using the training data of BraTS 2020, BraTS 2019 and BraTS 2018.  

BraTS 2020 https://www.med.upenn.edu/cbica/brats2020/data.html

BraTS 2019 https://www.med.upenn.edu/cbica/brats2019/data.html

BraTS 2018 https://www.med.upenn.edu/cbica/brats2018/data.html

## Prediction
We integrate data preprocessing, model and training in a python file.

After downloading the data, the file can be used to predict brain tumors.

To make predictions, run 1.0.py.

## Testing
Once you get the predictions, upload them in the CBICA portal to get the performance metrics.
