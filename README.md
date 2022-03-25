

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## Project Description
SVM and Convolutinal neural network have been 
trained on [EMNIST dataset](https://arxiv.org/pdf/1702.05373.pdf). The EMNIST data contains
images of hand writing, it consists of digits, [MNIST](http://yann.lecun.com/exdb/mnist/),
byclass, bymerge, and letters. 
The images in the dataset has shape of (28, 28, 1).
In the dataset, there are 731668, 82587 training and
testing byclass images respectively.
More dataset inforamtion in this [paper](https://arxiv.org/pdf/1702.05373.pdf).

![App Screenshot](https://github.com/faizan1234567/Hand-writing-recognition/blob/main/images/segmented_char5.png?raw=true)

Figure: 1 An (28, 28) image in EMNIST dataset

The dataset has been trained using [custom designed Convolutional neural networks](https://github.com/faizan1234567/Hand-writing-recognition/blob/main/training%26val%20files/models.ipynb).
The model shows 98% test accuracy on capital letters.

![App Screenshot](https://github.com/faizan1234567/Hand-writing-recognition/blob/main/images/model_detail.png?raw=true)

Figure: 2 A custom designed CNN model summary







## Installation

clone this repository

```bash
  !git clone https://github.com/faizan1234567/Hand-writing-recognition
  cd Hand-writing-recognition
```
Install import-ipynb package to get access to functions and data in 
a another notebook, and if you don't have the following packages please install it 
by using the following commands.
```bash
!pip install import-ipynb
!pip install torch
!pip install numpy
!pip install matplotlib
!pip install opencv-python
```


## Usage/Examples
Run ```OCR_model_Training_validation.ipynb``` to load data and preprocess the data,
furthermore, this notebook will import some function from 
```models.ipynb``` to load the customed designed CNN model. There are 
two cutstom desinged CNN models and one ANN model in this notebook. Moreover, 
```preprocess.ipynb``` contains code for image processing.

To Train the model, do the following.
```python
from models import ConvNet, ConvNet1, ANNmodel
model = ConvNet
n_epochs = 10  #num_epochs: play with it
b_size = 128   #batch size, please change it to see the results
learning_rate = 1e-3
num_classes = 10
training_dl, testing_dl = load_data(b_size) #data loaders: training data loader, testing data loader
run(model,training_dl, testing_dl, n_epochs, learning_rate)
```



## Demo
![App Screenshot](https://github.com/faizan1234567/Hand-writing-recognition/blob/main/images/origina_img.png?raw=true)

Figure: 3 An original test image
![App Screenshot](https://github.com/faizan1234567/Hand-writing-recognition/blob/main/images/output_img.png?raw=true) 

Figure: 4 Resulting image

![App Screenshot](https://github.com/faizan1234567/Hand-writing-recognition/blob/main/images/result1.png?raw=true)

Figure: 5 OCR output character recognition

![App Screenshot](https://github.com/faizan1234567/Hand-writing-recognition/blob/main/images/result.png?raw=true)

Figure: 6 capital letters recognition
## License

[MIT](https://choosealicense.com/licenses/mit/)

