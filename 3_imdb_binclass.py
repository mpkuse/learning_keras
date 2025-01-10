from tensorflow.keras.datasets import imdb 
from tensorflow.keras import layers 
from tensorflow import keras 
import numpy as np 
import random


import matplotlib.pyplot as plt

def plot_training_history(history):
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


class DecoderClass:
    def __init__(self): 
        self.wordIdx = imdb.get_word_index()
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
        

def Vectorize( xdata, dim=10000 ):
    # Given a list of (list of word indices). Returns a one-hot representation.  
    result = np.zeros( ( len(xdata), dim), dtype=np.float32 )

    for i in range( len( xdata  )):
        for wordIdx in  xdata[i] :
            # print( xdata[i][j], end=' ')
            if wordIdx > 2 and wordIdx<dim+3: 
                result[ i, wordIdx-3] = 1.0
        # print( "")
    return result 


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words=10000 )

    # Print some data to get a feel for it. 
    print( 'train_data: ', train_data.shape )
    print( 'train_labels: ', train_labels.shape )
    print( 'test_data: ', test_data.shape )
    print( 'test_labels: ', test_labels.shape )
    decoder = DecoderClass()
    for i in range(5):
        print( 'sample#', i, ' of total samples=',  len(train_data) ,'-> len=', len(train_data[i]), ' label=', train_labels[i]  )
        decoder.decode( train_data[i] )


    # Vectorize data for size: nSamples x 10000
    train_data_vec = Vectorize( train_data )
    train_label_vec = np.asarray( train_labels ).astype( 'float32' )
    test_data_vec = Vectorize( test_data )
    test_label_vec = np.asarray( test_labels ).astype( 'float32' )

    # Make a simple neural net 
    model = keras.Sequential( [ layers.Dense( 16, activation='relu'),  
                               layers.Dense( 16, activation='relu'), 
                               layers.Dense( 1, activation='sigmoid') ] )
    
    # Fit the data 
    model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
    model.summary()
    history = model.fit(train_data_vec,  train_label_vec, epochs=5, batch_size=512, validation_data=( test_data_vec[:1000], test_label_vec[:1000]) )
    plot_training_history( history )
    

    # Test: 
    test_loss, test_acc = model.evaluate( test_data_vec, test_label_vec )
    print( "test_loss: ", test_loss )     
    print( "test_acc: ", test_acc )

    # Some sample prediction 
    for _ in range(5):  # Loop 5 times
        random_number = random.randint(0, len( test_data_vec ) )

        print( '---\nsample#', random_number, ' of total samples=',  len(test_data_vec) ,'-> len=', len(test_data[random_number]), ' label=', test_labels[random_number]  )
        decoder.decode( test_data[random_number] )

        test_vec = test_data_vec[ random_number, : ].reshape( 1, -1 )
        print( 'Prediction:> ', model.predict( test_vec  ), '   Expected Outcome:> ', test_labels[ random_number] )


    
    exit()
    
    
    
