# Indian-Sign-Language-Recognition

Sign Languages are a set of languages that use predefined actions and movements to convey a message. These languages are primarily developed to aid deaf and other verbally challenged people. They use a simultaneous and precise combination of movement of hands, orientation of hands, hand shapes etc. Different regions have different sign languages like American Sign Language, Indian Sign Language etc. The focus here is on Indian Sign language .

In this project, the aim is towards designing and developing an Indian Sign Language recognition system that would help any deaf and dumb person commnicate to a perfectly abled person without the need of a translator, and vice-versa.

The purpose of this project is to recognize all the alphabets (A-Z) and digits (0-9) of Indian sign language using bag of visual words model and convert them to text/speech. Dual mode of recognition is implemented for better results. TDifferent machine learning techniques like Support Vector Machines (SVM), Logistic Regression, K-nearest neighbors (KNN) and a neural network technique Convolution Neural Networks (CNN), are explored for detection of sign language. The dataset for this system is created manually in different hand orientations and a train-test ratio of 80:20 is used.

## Getting Started

### Pre-requisites

Before running this project, make sure you have following dependencies -

- [pip](https://pypi.python.org/pypi/pip)
- [Python 3.7.1](https://www.python.org/downloads/)
- [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)
- [Opencv contrib](https://pypi.org/project/opencv-contrib-python/)

### Dataset

Some images of the dataset are shown below:

<p align="center">
  <br>
<img align="center" src="https://github.com/bhanu-1902/Indian-Sign-Language/blob/master/Images/dataset.png" width="800" height="750"> 
 </p>

Now, using `pip install` command, include the following dependencies

- Numpy
- Pandas
- Sklearn
- Tensorflow
- Scipy
- Keras
- Opencv
- Tkinter
- Sqlite3
- Pyttsx3
- SpeechRecognition (Google speech API)

## Workflow

### Preprocessing

Here 2 methods for preprocessing are used. First one is the background subtraction using an additive method, in which the first 30 frames are considered as background and any new object in the frame is then filtered out. Second one uses the skin segmentation concept, which is based on the extraction of skin color pixels of the user.

<p align="center">
  <br>
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/mask.png">       <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/after mask.png">       <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/canny.png">
  <br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Mask &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;After applying mask &nbsp; &nbsp; &nbsp;&nbsp;Canny Edge detection
</p>
  <br>
  
### Feature Detction and Extraction:
The Speeded Up Robust Feature (SURF) technique is used to extract descriptors from the segmented hand gesture images. These descriptors are then clustered to form the similar clusters and then the histograms of visual words are generated, where each image is represented by the frequency of occurrence of all the clustered features. The total classes are 36.
<p align="center">
  <br>
  <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/SURF.png">
 <br>
 &nbsp&nbsp&nbsp&nbsp&nbsp SURF Features
</p>

### Classification

The SURF descriptors extracted from each image are different in number with the same dimension (64). However, a multiclass SVM requires uniform dimensions of feature vector as its input. Bag of Features (BoF) is therefore implemented to represent the features in histogram of visual vocabulary rather than the features as proposed. The descriptors extracted are first quantized into 150 clusters using K-means clustering. Given a set of descriptors, where K-means clustering categorizes numbers of descriptors into K numbers of cluster center.

The clustered features then form the visual vocabulary where each feature corresponds to an individual sign language gesture. With the visual vocabulary, each image is represented by the frequency of occurrence of all clustered features. BoF represents each image as a histogram of features, in this case the histogram of 24 classes of sign languages gestures.

### Classifiers

After obtaining the baf of features model, we are set to predict results for new raw images to test our model. Following classifiers are used :

- Naive Bayes
- Logistic Regression classifier
- K-Nearest Neighbours
- Support Vector Machines
- Convolution Neaural Network

### Output

The predicted labels are shown in the form of text as well as speech using the python text to speech conversion library, Pyttsx3.

### Credits

- [Bag of Visual Words (BOVW)](https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f)
- [Image Classification with Convolutional Neural Networks](https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8)
