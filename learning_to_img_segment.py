import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.utils import plot_model

import random 
import os 

class PairedImageDataGenerator(tf.keras.utils.Sequence):
    def interpret_file_list( self, file_with_imagelist, max_items=-1, prefix="", postfix=".jpg" ):
        first_column = []
        random.seed(42)

        # n = 0
        with open(file_with_imagelist, 'r') as file:
            for line in file:
                # Split the line by whitespace and take the first element
                first_column.append( prefix + line.split()[0] + postfix )
                # n+= 1
                # if n == max_items:
                #     break 

        # return 

        if max_items <= 0:
            return first_column
        else: 
            return random.sample(first_column, max_items)

    def __init__(self, db_base, file_with_imagelist, batch_size, max_items, target_size=(224, 224)): 
        super().__init__()
        self.input_files = self.interpret_file_list( db_base+"/"+file_with_imagelist, max_items, prefix=db_base+"/images/", postfix=".jpg" )
        self.target_files = self.interpret_file_list( db_base+"/"+file_with_imagelist, max_items, prefix=db_base+"/trimaps/", postfix=".png" )
        self.batch_size = batch_size
        self.target_size = target_size
        self.indexes = np.arange(len(self.input_files))  # Index array for shuffling

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.input_files) / self.batch_size))

    def __getitem__(self, index):
        # Get batch file paths
        input_batch_files = self.input_files[index * self.batch_size:(index + 1) * self.batch_size]
        target_batch_files = self.target_files[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize batch arrays
        input_images = []
        target_images = []
        
        for input_path, target_path in zip(input_batch_files, target_batch_files):
            # Load and preprocess the input image
            input_img = load_img(input_path, target_size=self.target_size)
            input_img = img_to_array(input_img) #/ 255.0  # Normalize to [0, 1]
            input_images.append(input_img)
            
            # Load and preprocess the target image
            target_img = load_img(target_path, target_size=self.target_size, color_mode='grayscale')
            # target_img = load_img(target_path, target_size=self.target_size)
            target_img = img_to_array(target_img) - 1 
            target_images.append(target_img)

        return np.array(input_images), np.squeeze(  np.array(target_images), axis=-1 )

    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch
        np.random.shuffle(self.indexes)
        self.input_files = [self.input_files[i] for i in self.indexes]
        self.target_files = [self.target_files[i] for i in self.indexes]


if __name__ == '__main__':

    target_size = (256, 256)
    log_dir = "./8_learning_to_img_segment_fulltrainingset_tensorboard"
    os.mkdir( log_dir )
    # os.makedirs(log_dir, exist_ok=True)
    train_dataset = PairedImageDataGenerator(  '/home/mpkuse/Downloads/oxford-iiit/', 'train.txt', 8, -1, target_size=target_size )
    validation_dataset = PairedImageDataGenerator(  '/home/mpkuse/Downloads/oxford-iiit/', 'trainval.txt', 8, 200, target_size=target_size )

    ## Inspect data
    if False: 
        batch_idx = 4
        instance_idx = 30
        batch_input, batch_target = train_dataset[batch_idx]  # Get the first batch

        print("Input batch shape:", batch_input.shape)  # Shape of input images
        print("Target batch shape:", batch_target.shape)  # Shape of target images
        print( "Total Batches: ", len(train_dataset) )
        print("Total items in generator:", len(train_dataset.input_files) )

        # Visualize the first input-target pair
        plt.figure(figsize=(10, 5))

        # Input image
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(batch_input[instance_idx].astype( np.uint8) )  # Display the first input image
        plt.axis('off')

        # Target image
        plt.subplot(1, 2, 2)
        plt.title("Target Image")
        plt.imshow(batch_target[instance_idx])  # Display the corresponding target image
        plt.axis('off')

        plt.show()


    ## Make Model 
    inputs = keras.Input( shape=(target_size[0],target_size[1],3) )

    data_augmentation = keras.Sequential( [ layers.RandomFlip( "horizontal"),
                                        layers.RandomRotation(0.1),
                                            layers.RandomZoom( 0.2 ) ] )

    # x = data_augmentation( inputs )
    x = inputs
    x = layers.Rescaling( 1./255 )( x )

    x = layers.Conv2D( 64, 3, strides=2, activation="relu", padding="same" )( x )
    x = layers.Conv2D( 64, 3,  activation="relu", padding="same" )( x )
    x = layers.Conv2D( 128, 3, strides=2, activation="relu", padding="same" )( x )
    x = layers.Conv2D( 128, 3, activation="relu", padding="same" )( x )
    x = layers.Conv2D( 256, 3, strides=2, activation="relu", padding="same" )( x )
    x = layers.Conv2D( 256, 3, activation="relu", padding="same" )( x )

    x = layers.Conv2DTranspose( 256, 3, activation="relu", padding="same" )( x )
    x = layers.Conv2DTranspose( 256, 3, strides=2, activation="relu", padding="same" )( x )
    x = layers.Conv2DTranspose( 128, 3, activation="relu", padding="same" )( x )
    x = layers.Conv2DTranspose( 128, 3, strides=2, activation="relu", padding="same" )( x )
    x = layers.Conv2DTranspose( 64, 3, activation="relu", padding="same" )( x )
    x = layers.Conv2DTranspose( 64, 3, strides=2, activation="relu", padding="same" )( x )

    outputs = layers.Conv2D( 3, 3, activation="softmax", padding="same" )( x )

    model = keras.Model( inputs=inputs, outputs=outputs )
    model.summary()
    # plot_model(model, to_file=log_dir+'/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=32)


    ## Train Model 
    model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    tensorboard_cb = keras.callbacks.TensorBoard( log_dir=log_dir )
    checkpoint_cb = keras.callbacks.ModelCheckpoint( filepath=log_dir+"/convnet_from_scratch.keras", save_best_only=True, monitor="val_loss" )
    history = model.fit( train_dataset, epochs=50, validation_data=validation_dataset,  callbacks=[ tensorboard_cb, checkpoint_cb ] )

    breakpoint()