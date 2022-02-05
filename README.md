# Hand-writing-recognition
In this repo, SVM and Convolutinal neural network have been trained on [EMNIST dataset](https://arxiv.org/pdf/1702.05373.pdf). The EMNIST data contains images of hand writing, it consists of digits, [MNIST](http://yann.lecun.com/exdb/mnist/), byclass, bymerge, and letters. The images in the dataset has shape of (28, 28) with 1 channel (grayscale). In the dataset, there are  731,668, 82,587 training and testing byclass images respectively. More inforamtion in this [paper](https://arxiv.org/pdf/1702.05373.pdf).

![emnist_images](https://user-images.githubusercontent.com/61932757/152636586-7bd3ed5b-54d4-4c50-b70a-dacd77582fc2.png)

Figure1: an image in EMNIST dataset

The dataset is large, I experimented with byclass, digits and letters. The letters dataset has been trained using a custom designed Convolutional neural network and SVM for comparsion purposes. The model has acheived 90% test accuracy on letters dataset and 78% when trained using support vector machines. When it trained on upper case lettes it achieved 98% test accuracy. The model confuses some upper case letters with lower cases letters when trained on byclass category. 

![model_detail](https://user-images.githubusercontent.com/61932757/152636619-f0952d8c-09a0-43b4-9fe2-b0e5f16ff2da.png)

Figure2: A custom desinged CNN model

In the evaluation stage, a upper case letter hand writing image has been used. The image is processed through image processing stage, which binarize and segments an image into characters by find contours method. These images are then resized to (28, 28). Those small characters are then fed to the model for prediction. Modle prediction then plotted on a bounding box an original image.

![segmented_char](https://user-images.githubusercontent.com/61932757/152636638-24322c96-9380-4418-ac6b-25f58bb1cb6b.png) ![segmented_cahr6](https://user-images.githubusercontent.com/61932757/152636642-a89c0e43-e736-4dee-8f5a-0e4a9b4702dc.png)

Figure3: Segmented characters in a test image

![origina_img](https://user-images.githubusercontent.com/61932757/152636656-e50739e2-52be-44e5-82b7-84748f41a5ea.png)

Figure4: original test image

![output_img](https://user-images.githubusercontent.com/61932757/152636660-1eae0520-6c24-4274-b08c-0a5b886b2201.png)

Figure5: resulting image

![result](https://user-images.githubusercontent.com/61932757/152636672-c694da64-b6ee-46eb-945d-16389d867027.png) ![result1](https://user-images.githubusercontent.com/61932757/152636676-098c93da-cc66-45a1-bdfd-99022e7ee7dd.png)

Figure6: results on other images 
