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