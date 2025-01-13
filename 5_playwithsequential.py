from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow import keras 
import numpy as np 

def SequentialAPI(): 
    # Create a Sequential model
    model = Sequential()

    # Add layers to the model
    # model.add( Input(shape=(3, )) )
    model.add(Dense(64, input_dim=10, activation='relu'))  # Input layer
    model.add(Dense(32, activation='relu'))               # Hidden layer
    model.add(Dense(2, activation='sigmoid'))             # Output layer

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()
    breakpoint()


def FunctionalAPI():
    inputs = keras.Input( shape=(5,), name="my_input" )
    features = keras.layers.Dense( 64, activation="relu")( inputs )
    output = keras.layers.Dense( 10, activation="softmax")( features )
    model = keras.Model( inputs=inputs, outputs=output )

    # Summary of the model
    model.summary()
    keras.utils.plot_model( model, "FunctionalAPI_1.png", show_shapes=True)
    breakpoint()

def FunctionalAPIComplicated():
    n_vocab = 10000
    n_tags = 100 
    input_title = keras.Input( shape=(n_vocab,), name="input_title" )
    input_text = keras.Input( shape=(n_vocab,), name="input_text" )
    input_tags = keras.Input( shape=(n_tags,), name="input_tags" )

    inputs = [input_title, input_text, input_tags] 

    features = keras.layers.Concatenate()( inputs )
    features = keras.layers.Dense( 64, activation='relu')( features )
    features = keras.layers.Dense( 128, activation='relu')( features )

    output_priority = keras.layers.Dense( 1, activation='sigmoid'  )( features )
    output_department = keras.layers.Dense( 7, activation='softmax' )( features )

    model = keras.Model( inputs,  outputs=[output_priority, output_department])
    keras.utils.plot_model( model, "FunctionalAPIComplicated.png", show_shapes=True, show_layer_names=True)

    # Generate some data for training 
    n_samples = 4096
    title_data = np.random.randint( 0, 2, size=(n_samples,n_vocab) )
    text_data = np.random.randint( 0, 2, size=(n_samples, n_vocab) )
    tags_data = np.random.randint( 0, 2, size=(n_samples, n_tags ) )
    priority_data = np.random.random( size=(n_samples, 1 ) )
    department_data = np.random.randint( 0,2 , size=(n_samples, 7 ) )

    model.compile( optimizer='rmsprop', loss=['mean_squared_error', 'categorical_crossentropy'], metrics=[ ['mean_absolute_error'], ['accuracy'] ] )

    model.fit( [title_data, text_data, tags_data ], [priority_data, department_data ], epochs=5 )


# SequentialAPI()
# FunctionalAPI()
FunctionalAPIComplicated()