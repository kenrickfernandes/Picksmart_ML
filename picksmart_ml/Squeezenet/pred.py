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
#from mobile import MobileNetv2
#import Efficientnet
#from keras import models
import tensorflow as tf
import argparse
from sklearn.metrics import classification_report, confusion_matrix



train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="/test/brampton_workspace/214_class_train/",
    target_size=(227, 227)
)
test_generator = train_datagen.flow_from_directory(
   # directory="test_crop/classification_test_crops_40079/",
    directory = "/test/brampton_workspace/202_class_test/",
    batch_size = 1,
    shuffle = False,
    target_size=(227, 227)
)
print(train_generator.class_indices)
print(test_generator.class_indices)
#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
np.savetxt("214_class_list.txt",np.array(list(train_generator.class_indices.keys())),fmt='%s')
print("saved class list!!!!!!!!!!!!!!")
model = keras.models.load_model("ckpt_214/model_train.18.h5")

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
#print(np.shape(pred))
#print(type(pred[0]))
a = np.array(pred)
np.savetxt("foo_214_sigmoid.csv", a, delimiter=",")
#print(test_generator.classes)
predicted_class_indices=np.argmax(pred,axis=1)
predicted_class_prob=list(np.max(pred,axis=1))
print(train_generator.class_indices)
print(test_generator.class_indices)
print(confusion_matrix(test_generator.classes, predicted_class_indices))
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
#e=confusion_matrix(test_genrator.classes,predicted_class_indices)
#np.savetxt("confusion_matrix_sigmoid.csv", e, delimiter=",")
#clsf_report = pd.DataFrame(classification_report(test_generator.classes, y_pred = pre, output_dict=True)).transpose()
#clsf_report.to_csv('Your Classification Report Name.csv', index= True)
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
    "Predictions":predictions,"Confidence":predicted_class_prob})
results.to_csv("results_214_sigmoid_with_confidence.csv",index=False)


def main():
    parser = argparse.ArgumentParser(description='Squeezenet ')
    parser.add_argument("--train_data",help="Train Data Directory")
    parser.add_argument("--test_data",help="Test Data Directory")
    parser.add_argument("--model_checkpoint",help="Path to Model Weights .h5")
    parser.add_argument("--results_out",help="Path to save results")
    args = vars(parser.parse_args())



if __name__ == '__main__':
    main()