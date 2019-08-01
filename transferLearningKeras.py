#NOTE: on colab, we must use "!" to install or run like cli.
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

#install framework
!pip install keras
!pip install onnxmltools
!pip install winmltools

#Copy dataset from drive to current place
!cp "/content/drive/My Drive/dataset.zip" . 

#Unzip dataset
!unzip dataset.zip

#If content of dataset unzipped on outside. We have to make a folder and give it to this one.
!mkdir dataset
!mv [nameFile] dataset

#Import library
import sys
import shutil
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
import numpy as np

from onnxmltools.utils import save_model
import winmltools
import tf2onnx
import onnxmltools

#make a folder training a save result
os.makedirs("training", exist_ok=True)
os.makedirs("training/training", exist_ok=True)
os.makedirs("training/validation", exist_ok=True)

list_classes = os.listdir("dataset")
split_rate = 0.7
for label in list_classes:
  os.makedirs("training/training/"+label,exist_ok=True)
  os.makedirs("training/validation/"+label,exist_ok=True)
  
  files = os.listdir("dataset/"+label)
  training_num = int(len(files)*split_rate)
  for i in range(training_num):
    shutil.move("dataset/"+label+"/"+files[i],"training/training/"+label+"/"+files[i])
  for i in range(training_num,len(files)):
    shutil.move("dataset/"+label+"/"+files[i],"training/validation/"+label+"/"+files[i])

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)
# hyper parameters for model
nb_classes = 10  # number of classes 
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 224, 224  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation

#Define model to train with? I use transfer learning with Keras (Xception)
def train(train_data_dir, validation_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Top Model Block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())

    # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
    # # uncomment when choosing based_model_last_block_layer
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio,
                                       shear_range=transformation_ratio,
                                       zoom_range=transformation_ratio,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)
                                     # validation_split=0.3) split automatic (when i use model with 49 classes)

    #validation_datagen = ImageDataGenerator(rescale=1. / 255)

    #os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
                                                     #  subset="training") subset I've added when I train with 49 classes
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = train_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                 subset="validation")

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc

    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5') #If u wanna save all (model json file, top_weights, ...), u choose save all

    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=0) #use EarlyStopping to stop when model does not increas in future
    ]

    # Train Simple CNN
    model.fit_generator(train_generator,
                        samples_per_epoch=7000,
                        nb_epoch=nb_epoch / 5,
                        validation_data=validation_generator,
                        nb_val_samples=3000,
                        callbacks=callbacks_list)
    # verbose
    print("\nStarting to Fine Tune Model\n")

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    model.load_weights(top_weights_path)

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc
    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    ]

    # fine-tune the model
    model.fit_generator(train_generator,
                        samples_per_epoch=7000,
                        nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=3000,
                        callbacks=callbacks_list)

    # save model
   
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    return model

data_dir = os.path.abspath("training")
train_dir = os.path.join(os.path.abspath(data_dir), 'training')  # Inside, each class should have it's own folder
validation_dir = os.path.join(os.path.abspath(data_dir), 'validation')  # each class should have it's own folder
model_dir = os.path.abspath("models")

os.makedirs(os.path.join(os.path.abspath(data_dir), 'preview'), exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

trained_model = train(train_dir, validation_dir, model_dir)  # train model

trained_model.save_weights("saved_model.h5") #save model H5 in keras
onnx = onnxmltools.convert_keras(trained_model)
onnxmltools.save_model(onnx, "converted.onnx")

    # release memory
k.clear_session()