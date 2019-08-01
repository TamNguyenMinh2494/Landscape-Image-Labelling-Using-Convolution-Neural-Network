# LANDSCAPE IMAGE LABELLING USING CONVOLUTION NEURAL NETWORK

---

## Content 

[1. About Me](#about-me)

[2. Why I choose this topic?](#why)

[3. How do I collect data?](#how)

[4. Methodology](#methodology)

[5. Result](#result)

[6. Deploy on GCP](#deploy)


[7. Summary](#summary)

[8. Reference](#reference)

[9. Acknowledgement](#acknowledgement)

## 1.About Me <a name="#about-me"></a>

Hello guys, welcome to my github. My name is ***Nguyen Minh Tam***. Currently, I am a junior at Hoa Sen University, VietNam. I have come to Machine Learning Camp to do my project and meet new friends on other countries. Here, I have memories, experiences, friends that will never be forgotten.

## 2. Why I choose this topic? <a name="#why"></a>

During the rapid urbanization process, the land use management is necessary and inevitable. In fact, the land use management is the work of the governments. However, to enhance accuracy, and quickly update changes, it is necessary to apply modern technologies such as Big Data, Machine Learning (ML) / Deep Learning (DL). Moreover, this must be a source of data for those who want to buy and sell land for reference, to understand government regulations and to protect their interests. The most important issue is to build an automated process that helps people share information about their land or they know. At the same time, processing and storing data is shared automatically to help governments and everyone. So, that are the reason I chose this topic.

## 3. How do I collect data? <a name="#how"></a>

Because data is also quite large, I have a data collection solution from the collaborators (crowdsource) by building an application for people to take photos and store on it. In addition, I also took photos from the internet and around my home. As a result, I have got the dataset set with 10 classes and more than 10 thousand images. 

## 4. Methodology <a name="#methodology"></a>

After reseached about machine learning and libraries, I refer use transfer learning with **Xception** model and Python programming language for my project. This model gets to a top-1 validation accuracy. **Nesterov Adam** optimizer and categorical_crossentropy is loss function. 

After training with the model, I realized that concluding that image belongs to a class is not feasible, because the image can exist more than 1 layer. For instance, you can see the image below. 

![alt text](https://icdn.dantri.com.vn/thumb_w/640/2018/4/29/duong-pho-vang-ve-3-15249707929941964684660.jpg)

We cannot conclude that it belongs to the urban land because we can also see the road in the image. Therefore, in order to evaluate this photo, we can assume that in this photo, it is both urban land and transportation land, but in proportion, the urban land will occupy more. So, the project is divided into two parts for processing to increase accuracy.

*[Convolutional neural network](#cnn)

This is a model to classify input images. I have deleted unneccesary images and just keep correct images. I use dataset with 10 classes to train the model. For each input image, CNN will try to extract the characteristics of each image. The feature is called characteristic when we can use it to distinguish it from other photos. We do not try to specify which characteristics will be selected but will let CNN automatically perform through training. After filtering, the characteristics will be passed through the neural network layer. Finally, CNN will indicate what the input is likely to be according to the probability distribution for each class.

*[Object detection](#objDetect)

According to result I have after trainning with CNN, I will use Faster R-CNN to detect the characteristics in image one by one, so that I can calculate the ratio and number of occurrences of the object in the image. Combined with results from CNN, using algorithms, I can determine the rate at which class the photo belongs to.

## 5. Result <a name="#result"></a>

![result](/img/gcp_res.PNG)

## 6. Deploy on GCP <a name="#deploy"></a>

1. Create VM instance on GCP
![create-GCP](/img/create.PNG)
2. Configure the zone, CPU, disk, ...
![modify](/img/modify.jpg)
3. SSH
![ssh](/img/ssh_gcp.PNG)
4. Update/Upgrade

    ```
    sudo apt-get update
    ```


    ```
    sudo apt-get upgrade
    ```
5. Install openCV 
    ```
    sudo apt install python-opencv
    ```
6. Install python3.6 and pip3.6 or later.
7. Install tensorflow, keras, opencv into pip3.6
    ```
    sudo pip3.6 install tensorflow
    ```
    ```
    sudo pip3.6 install keras
    ```
    ```
    sudo pip3.6 install opencv-python
    ```
8. Upload top_model_weights.h5
+ We can use https://lutzroeder.github.io/netron/ to upload and see your model.
9. Upload script.py 
```python
    import cv2
import numpy as np
from keras import *

#load top model weight
inference_model = models.load_model("top_model_weights.h5")
#wget if using image on google or upload file on GCP
image_input = cv2.imread("test.jpg")
#resize input image
image_input = cv2.resize(image_input,dsize=(224,224))
#expand the dimension to array 
image_input = np.expand_dims(np.asarray(image_input), axis=0)
#predict
preds = inference_model.predict(image_input)
#add labels according to dataset
label_map = ["BCS","CHN","CLN","DGT","ODT","SKK","SON","TSN"]
i=0
data = ""
for pred in preds[0]:
  data +=("%s:%.4f\n"%(label_map[i],pred))
  i+=1
result = open('result.txt','w')
result.write(data)
result.close()
```
10. Upload picture
11. Run script
    ```
    python3.6 script.py
    ```
12. cat result
![cat_result](/img/gcp_ls.PNG)

## 7. Summary <a name="#summary"></a>

After all, this project has completed about 75%. In the camp, I had more experience working with Machine Learning and learning how to self-study effectively. Moreover, I have new friends from other countries, so we can share our knowledge and experience and have weekends together. Camp really worthy and interesting.

## 8. Reference <a name="#reference"></a>

1. CNN: 
* https://keras.io/applications/#xception
* https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn
* https://keras.io/


2. Object detection: 
* https://github.com/kbardool/keras-frcnn 
* https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/?source=post_page
* https://medium.com/@fractaldle/brief-overview-on-object-detection-algorithms-ec516929be93

## 9. Acknowledgement <a name="#acknowledgement"></a>

This project will not be completed without an effective working environment from the support of ***Jeju National University and Jeju Developement Center and sponsors***. I would like to express my sincere thanks to **Prof. Yungcheol Byun** and mentor **Dr.Lap Nguyen Trung** for guided and helped me in this camp.