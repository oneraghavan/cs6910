import argparse
import os
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

import wandb
from wandb.keras import WandbCallback

MODELS_TO_TRY = {
  "IV2" : InceptionResNetV2,
  "IV3" : InceptionV3,
  "resnet" : ResNet50,
  "Xception" : Xception
}
no_of_classes = 10

def load_model(model_name, fc,args):
  """Loading pretrain model """

  model_type = MODELS_TO_TRY[model_name]
  model = model_type(input_shape=(600,800,3),weights='imagenet', include_top=False) ##here (600,800) is the inaturalist image dimension 
  x = model.output
  x = GlobalAveragePooling2D()(x)
  x = layers.Dense(fc, activation='relu')(x) ## Modifying the fully connected layer
  x = layers.Dropout(args.dropout)(x)                  
  op = layers.Dense(no_of_classes, activation='softmax')(x) ##here 10 is the number of classes
  model = Model(inputs=model.input, outputs=op)
  
  # freeze all  layers
  for layer in model.layers:
    layer.trainable = False
  
  model.compile(optimizer=RMSprop(lr=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def wandb_config_update(model, wandb_config, args):
  """Logging model parameters to wandb"""

  wandb_config.update({
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "train_examples" : args.num_train,
    "valid_examples" : args.num_valid,
    "fc" : args.fc,
    "pre_epochs" : args.pretrain_epochs,
    "lr" : args.learning_rate,
    "mnt" : args.momentum,
    "freeze_layer" : args.freeze_layer,
    "model_type" : args.model_type 
  })


def fit_model(args):

  wandb.init(project=args.project_name,entity=args.entity)
  callbacks = [WandbCallback()]
  

  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  train_dir = os.path.join(args.root_dir, 'train')
  validation_dir = os.path.join(args.root_dir, 'val')


  train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = args.batch_size,
                                                    class_mode = 'categorical', 
                                                    target_size = (600, 800))   

  validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                          batch_size  = args.batch_size,
                                                          class_mode  = 'categorical', 
                                                          target_size = (600, 800))

  model = load_model(args.model_type, args.fc,args)
  wandb_config_update(model, wandb.config, args)

#   ##pretraining the models
  model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.pretrain_epochs,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  
  for layer in model.layers[:args.freeze_layer]:
    layer.trainable = False
    print(layer.name + "Frozen")

  for layer in model.layers[args.freeze_layer:]:
    layer.trainable = True
    print(layer.name + "re-trained")

  model.compile(optimizer=optimizers.SGD(lr=args.learning_rate, momentum=args.momentum), loss='categorical_crossentropy', metrics=["accuracy"])
  model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.epochs,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-p",
    "--project_name",
    type=str,
    default="Deep learning assignment",
    help="wandb project name")
  
  parser.add_argument(
      "--entity",
      type=str,
      default="sandhyabaskaran",
      help="Wandb entity")

  parser.add_argument(
    "--dropout",
    type=float,
    default=0.2,
    help="dropout rate")

  parser.add_argument(
    "--fc",
    type=int,
    default=1024,
    help="Change the fully connected layer to fc number of nodes")

  parser.add_argument(
    "-fl",
    "--freeze_layer",
    type=int,
    default=155,
    help="Freeze upto layer")

  parser.add_argument(
    "-m",
    "--model_type",
    type=str,
    default="IV3",
    help="The model type to load IV2,IV3,Xception or resnet")

  parser.add_argument(
    "--pretrain_epochs",
    type=int,
    default=5,
    help="Pretrain for this number of epochs")

  parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.0001,
    help="gradient descent learning rate")

  parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="GD momentum rate")

  parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size")

  parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Number of training epochs")

  parser.add_argument(
    "-nt",
    "--num_train",
    type=int,
    default=9999,
    help="Number of training examples. max (9999)")

  parser.add_argument(
    "-nv",
    "--num_valid",
    type=int,
    default=2000,
    help="Number of validation examples max(2000)") 

  parser.add_argument(
    "--root_dir",
    type=str,
    default="inaturalist_12K/",
    help="Absolute path of data directory")

 
  args = parser.parse_args()
  fit_model(args)
