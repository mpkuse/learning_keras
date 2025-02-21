import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.utils import plot_model

import random 
import os 

from learning_to_img_segment import PairedImageDataGenerator


target_size = (256, 256)
log_dir = "./8_learning_to_img_segment_tensorboard"
log_dir = "./8_learning_to_img_segment_fulltrainingset_tensorboard"
test_dataset = PairedImageDataGenerator(  '/home/mpkuse/Downloads/oxford-iiit/', 'train.txt', 8, 1000, target_size=target_size )



if True: 
    batch_idx = 3
    instance_idx = 1
    batch_input, batch_target = test_dataset[batch_idx]  # Get the first batch


    ## Load Model
    model = keras.models.load_model( log_dir + "/convnet_from_scratch.keras" ) 
    mask = model.predict(  batch_input )
    mask = np.argmax( mask, axis=-1 )
    print( 'mask.shape=', mask.shape)

    print("Input batch shape:", batch_input.shape)  # Shape of input images
    print("Target batch shape:", batch_target.shape)  # Shape of target images
    print( "Total Batches: ", len(test_dataset) )
    print("Total items in generator:", len(test_dataset.input_files) )

    # Visualize the first input-target pair
    plt.figure(figsize=(10, 5))

    # Input image
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(batch_input[instance_idx].astype( np.uint8) )  # Display the first input image
    plt.axis('off')

    # Target image
    plt.subplot(1, 3, 2)
    plt.title("Target Image")
    plt.imshow(batch_target[instance_idx])  # Display the corresponding target image
    plt.axis('off')

    # Prediction
    plt.subplot(1, 3, 3)
    plt.title("Predicted Image")
    plt.imshow(mask[instance_idx])  # Display the corresponding target image
    plt.axis('off')

    plt.show()

