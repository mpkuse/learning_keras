from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


#-- Load Data 
(tImg, tLabels), (testImg, testLabels) = mnist.load_data()
tImg = tImg.reshape( ( len(tImg), 28*28) ).astype( "float32") / 255
testImg = testImg.reshape( (len(testImg), 28*28) ).astype( "float32") / 255

#-- Build a neural network
model = keras.Sequential( [ layers.Dense( 512, activation="relu" ), layers.Dense( 10, activation="softmax") ] )


#-- Set optimizer 
model.compile( optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#-- Train
model.fit( tImg, tLabels, epochs=5, batch_size=128 )

model.summary()
config = model.get_config()
print(config)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

exit() 

#-- Predict 
prediction = model.predict( testImg[0:50] )
print( prediction.shape)
print( prediction )

for i in range(prediction.shape[0]):
    print( prediction[i].argmax(), "<-->", testLabels[i]  )
    

test_loss, test_acc = model.evaluate( testImg, testLabels )
print( "test_loss: ", test_loss )     
print( "test_acc: ", test_acc )