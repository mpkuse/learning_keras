from tensorflow.keras.datasets import imdb 
from tensorflow.keras import layers 
from tensorflow import keras 
import numpy as np 
import random

import matplotlib.pyplot as plt

def plotTrainingHistory(history):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    
    Parameters:
    - history: Keras History object returned by model.fit()
    """
    history_dict = history.history

    # Extract metrics
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']

    # Define the number of epochs
    epochs = range(1, len(loss) + 1)

    # Create the plots
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    plt.show()


def vectorizeIntegerList( xdata, dim=10000 ):
    # Given a list of (list of word indices). Returns a one-hot representation.  
    result = np.zeros( ( len(xdata), dim), dtype=np.float32 )

    for i in range( len( xdata  )):
        for wordIdx in  xdata[i] :
            # print( xdata[i][j], end=' ')
            if wordIdx > 2 and wordIdx<dim+3: 
                result[ i, wordIdx-3] = 1.0
        # print( "")
    return result 

def toOneHot( xlabels, dim ):
    result = np.zeros( (len(xlabels), dim), dtype=np.float32 )
    for i, l in enumerate(xlabels):
        result[ i, l ] = 1.0
    return result

class DecoderClass:
    def __init__(self, database_obj): 
        self.wordIdx = database_obj.get_word_index()
        # [ ..., 'word': 23, 'word2': 74, ... ]

        self.reverseWordIdx = {}
        for key,val in self.wordIdx.items():
            assert( val not in self.reverseWordIdx.keys() )
            self.reverseWordIdx[ val ] = key 

    def decode( self, wordidx_list ):
        # for wordidx in wordidx_list:
        #     print( wordidx, end=' ' )
        # print( )

        for wordidx in wordidx_list:
            if wordidx == 0: 
                print( "%", end=' ' )
            if wordidx == 2:
                print( "?", end=' ') 
            if wordidx > 2: 
                print(  self.reverseWordIdx[ wordidx-3 ], end=' ' )

        print( ' ')
        