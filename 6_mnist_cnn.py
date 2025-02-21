from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


#-- Load Data 
(tImg, tLabels), (testImg, testLabels) = mnist.load_data()
# tImg = tImg.reshape( ( len(tImg), 28*28) ).astype( "float32") / 255
# testImg = testImg.reshape( (len(testImg), 28*28) ).astype( "float32") / 255
tImg = tImg.astype( "float32") / 255
testImg = testImg.astype( "float32") / 255


#-- Build a neural network
# model = keras.Sequential( [ layers.Dense( 512, activation="relu" ), layers.Dense( 10, activation="softmax") ] )

inputs = keras.Input( shape=(28,28,1) )
x = layers.Conv2D( filters=32, kernel_size=3, activation="relu" )( inputs )
x = layers.MaxPooling2D( pool_size=2)( x )

x = layers.Conv2D( filters=64, kernel_size=3, activation="relu" )( x )
x = layers.MaxPooling2D( pool_size=2)( x )

x = layers.Conv2D( filters=128, kernel_size=3, activation="relu" )( x )
x = layers.Flatten()( x )

outputs = layers.Dense( 10, activation='softmax' )( x )
model = keras.Model( inputs=inputs, outputs=outputs )

keras.utils.plot_model( model, "FunctionalAPIComplicated.png", show_shapes=True, show_layer_names=True)
model.summary()

#-- Set optimizer 
model.compile( optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"] )

#-- Tensorboard 
tensorboard = keras.callbacks.TensorBoard( log_dir="./1_tensorboard" )

#-- Train
model.fit( tImg, tLabels, epochs=5, batch_size=64, validation_data=(testImg[0:1000], testLabels[0:1000]), callbacks=[ tensorboard ] )

model.summary()
config = model.get_config()
print(config)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# exit() 

#-- Predict 
prediction = model.predict( testImg[0:50] )
print( prediction.shape)
print( prediction )

for i in range(prediction.shape[0]):
    print( prediction[i].argmax(), "<-->", testLabels[i]  )
    

test_loss, test_acc = model.evaluate( testImg, testLabels )
print( "test_loss: ", test_loss )     
print( "test_acc: ", test_acc )