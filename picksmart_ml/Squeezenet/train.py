import pandas as pd
import os
import cv2
import numpy as np
import random
random.seed(32)
from tensorflow import keras
#from keras import backend as K
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.utils import multi_gpu_model
import Squeezenet
#import Squeezenet_2 as Squeezenet
#from mobile import MobileNetv2
#import Efficientnet
from tensorflow.keras import models
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="model to train")
ap.add_argument("-t", "--train_dir", required=True, help="train_directory")
#ap.add_argument("-v","--val_dir",required=True,help="val_directory")
ap.add_argument("-c", "--num_classes", required=True, help="number of class")
ap.add_argument("-G", "--GPU", required=True, help="number of gpus")
#ap.add_argument("-I", "--model_input_size", required=True, help="model_input_size like (227, 227)")
ap.add_argument("-b", "--train_batch_size", required=True, help="batch_size to train")
ap.add_argument("-e", "--epochs", required=True, help="number of epochs")
ap.add_argument("-s", "--input_size", required=True, help="input size to model")
ap.add_argument("-o","--output_checkpoint_folder",required=True,help="Folder to save Checkpoints")
#ap.add_argument("-csv", "--csv_name", required=True, help="classification csv name")

args = vars(ap.parse_args())
model_name = args["model"]
#args = vars(ap.parse_args())
train_dir = args["train_dir"]
#val_dir = args["val_dir"]
num_classes = int(args["num_classes"])
G = int(args["GPU"])
#model_input_size = args["model_input_size"]
train_batch_size = int(args["train_batch_size"])
epochs = int(args["epochs"])
image_size = int(args["input_size"])
#csv_name = args["csv_name"]

############################################## Pre-processing and model_input ###############################################

def fixed_rotation(img):
        angle_list=[0, 90, 180, 270]
        theta = random.choice(angle_list)
    #print("image shape",img.shape)
    #try:
        #print("sucess",img.shape)
        (image_size_height, image_size_width,_) = img.shape
        image_center_x = image_size_width // 2
        image_center_y = image_size_height // 2
        rotation = cv2.getRotationMatrix2D((image_center_x, image_center_y), theta, 1)
        rotated_img = cv2.warpAffine(img, rotation, (image_size_width, image_size_height))
        #print("sucess",img.shape)
        return rotated_img


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if model_name == "SqueezeNet":
        get_model = Squeezenet.SqueezeNet(nb_classes=num_classes)
        model = models.Sequential()
        model.add(get_model)

    train_datagenerator = ImageDataGenerator(rescale=1./255, preprocessing_function = fixed_rotation, horizontal_flip=True, validation_split=0.1)

    train_generator = train_datagenerator.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            class_mode = 'categorical',
            batch_size = train_batch_size,
            subset='training'
        )
    print(train_generator.class_indices)
    validation_generator = train_datagenerator.flow_from_directory(
            train_dir, # same directory as training data
            target_size=(image_size, image_size),
            batch_size = train_batch_size,
            subset='validation')
   #print(val_generator.class_indices)


    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
    print(class_weights)
    #print(type(class_weights))
    #class_weights_items = list(class_weights.items())
    #class_weights_array = np.array(class_weights_items)
    #print(class_weights_array)
    model.summary()

    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=6)
    mc_path = args['output_checkpoint_folder']
    mc = ModelCheckpoint(mc_path + '/model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    mc_2 = ModelCheckpoint(mc_path +'/model_train.{epoch:02d}.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.000000001)
    tensor_board = TensorBoard(log_dir='./logs')
    csv_logger = CSVLogger('training.log')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

#model = load_model('pallet_jack_17_class_squeezenet_model_9_apr.h5')

    model.fit(
            train_generator,
            steps_per_epoch = train_generator.samples // train_generator.batch_size,
            validation_data = validation_generator, 
            validation_steps = validation_generator.samples // validation_generator.batch_size,
            epochs = 1000,
                    callbacks=[mc, mc_2, reduce_lr, tensor_board, csv_logger],
                    class_weight=class_weights)

    #model.save('pallet_3.h5')
