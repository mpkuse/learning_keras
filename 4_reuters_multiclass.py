from tensorflow.keras.datasets import reuters 
from tensorflow.keras import layers 
from tensorflow import keras 
import numpy as np 
import random


from helpers import plotTrainingHistory
from helpers import DecoderClass
from helpers import vectorizeIntegerList, toOneHot



if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data( num_words=10000 )

    # Print some data to get a feel for it. 
    print( 'train_data: ', train_data.shape )
    print( 'train_labels: ', train_labels.shape )
    print( 'test_data: ', test_data.shape )
    print( 'test_labels: ', test_labels.shape )
    decoder = DecoderClass( reuters )
    for i in range(5):
        print( '----\nsample#', i, ' of total samples=',  len(train_data) ,'-> len=', len(train_data[i]), ' label=', train_labels[i]  )
        decoder.decode( train_data[i] )

    
    # Vectorize data for size: nSamples x 10000
    train_data_vec = vectorizeIntegerList( train_data )
    train_label_vec = toOneHot( train_labels, dim=46 ) #np.asarray( train_labels ).astype( 'float32' )
    test_data_vec = vectorizeIntegerList( test_data )
    test_label_vec = toOneHot( test_labels, dim=46 ) #np.asarray( test_labels ).astype( 'float32' )

    # Make a simple neural net 
    model = keras.Sequential( [ layers.Dense( 64, activation='relu'),  
                               layers.Dense( 64, activation='relu'), 
                               layers.Dense( 46, activation='sigmoid') ] )
    
    # Fit the data 
    model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    model.summary()
    history = model.fit(train_data_vec,  train_label_vec, epochs=20, batch_size=512, validation_data=( test_data_vec[:1000], test_label_vec[:1000]) )
    plotTrainingHistory( history )
    

    # Test: 
    test_loss, test_acc = model.evaluate( test_data_vec, test_label_vec )
    print( "test_loss: ", test_loss )     
    print( "test_acc: ", test_acc )
    exit() 

    # Some sample prediction 
    for _ in range(5):  # Loop 5 times
        random_number = random.randint(0, len( test_data_vec ) )

        print( '---\nsample#', random_number, ' of total samples=',  len(test_data_vec) ,'-> len=', len(test_data[random_number]), ' label=', test_labels[random_number]  )
        decoder.decode( test_data[random_number] )

        test_vec = test_data_vec[ random_number, : ].reshape( 1, -1 )
        print( 'Prediction:> ', model.predict( test_vec  ), '   Expected Outcome:> ', test_labels[ random_number] )

    exit()

    
    
    
    
