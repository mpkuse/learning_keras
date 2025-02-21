from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import plot_model

import os
import time 

log_dir = "./7_log_vgg_finetune_on_subset_tensorboard"
os.mkdir( log_dir )

base_dir = "/home/mpkuse/Downloads/kagglecatsanddogs_5340/PetImages/kaggle/subset_full"
base_dir = "/home/mpkuse/Downloads/kagglecatsanddogs_5340/PetImages/kaggle/subset"
train_dataset = image_dataset_from_directory( base_dir+"/train", 
                                             labels="inferred", 
                                             image_size=(180,180), 
                                             batch_size=32 )
validation_dataset = image_dataset_from_directory( base_dir+"/validation", 
                                            labels="inferred", 
                                            image_size=(180,180), 
                                            batch_size=32 )


inputs = keras.Input( shape=(180,180,3) )

data_augmentation = keras.Sequential( [ layers.RandomFlip( "horizontal"),
                                       layers.RandomRotation(0.1),
                                        layers.RandomZoom( 0.2 ) ] )
x = data_augmentation( inputs)

if False: 
    # Self created conv net 
    x = layers.Rescaling( 1./255 )( x )
    x = layers.Conv2D( filters=32, kernel_size=3, activation="relu" )( x )
    x = layers.MaxPooling2D( pool_size=2)( x )

    x = layers.Conv2D( filters=64, kernel_size=3, activation="relu" )( x )
    x = layers.MaxPooling2D( pool_size=2)( x )
    x = layers.Conv2D( filters=128, kernel_size=3, activation="relu" )( x )
    x = layers.MaxPooling2D( pool_size=2)( x )
    x = layers.Conv2D( filters=256, kernel_size=3, activation="relu" )( x )
    x = layers.MaxPooling2D( pool_size=2)( x )
    x = layers.Conv2D( filters=256, kernel_size=3, activation="relu" )( x )
    # x = layers.MaxPooling2D( pool_size=2)( x )

    x = layers.Flatten()( x )
    outputs = layers.Dense( 1, activation='sigmoid' )( x )

if True: 
    x = keras.applications.vgg16.preprocess_input(x)
    vgg_base = keras.applications.vgg16.VGG16( weights='imagenet', include_top=False )
    vgg_base.trainable = False
    x = vgg_base( x )
    x = layers.Flatten()( x )
    x = layers.Dense(256)(x)
    x = layers.Dropout( 0.5 )( x )
    outputs = layers.Dense( 1, activation="sigmoid" )( x )

model = keras.Model( inputs=inputs, outputs=outputs )
model.summary()

model.compile( loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

tensorboard_cb = keras.callbacks.TensorBoard( log_dir=log_dir )
checkpoint_cb = keras.callbacks.ModelCheckpoint( filepath=log_dir+"/convnet_from_scratch.keras", save_best_only=True, monitor="val_loss" )

with open(log_dir+'/model.summary.txt', "w") as f:
    # Redirect the model.summary() output to the file
    model.summary(print_fn=lambda x: f.write(x + "\n"))
# config = model.get_config()
# print(config)
plot_model(model, to_file=log_dir+'/model.png', show_shapes=True, show_layer_names=True)



history = model.fit( train_dataset, epochs=30, validation_data=validation_dataset,  callbacks=[ tensorboard_cb, checkpoint_cb ] )