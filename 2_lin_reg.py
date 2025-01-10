from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

import numpy as np 

import plotly.graph_objects as go


#-- Make Data 
p_feat = np.random.multivariate_normal( mean=[0,3], cov=[ [1,0.5], [0.5, 1] ], size=1000 )
n_feat = np.random.multivariate_normal( mean=[3,0], cov=[ [1,0.5], [0.5, 1] ], size=1000 )

p_labels = np.ones( 1000 )
n_labels = np.zeros( 1000 )

feat = np.concatenate( [p_feat, n_feat] ) # Nx2
labels = np.concatenate( [p_labels, n_labels]) #Nx1


#-- Build r = X*feat + b 
model = keras.Sequential( [ layers.Dense( 1, activation=None ) ] )

#-- Compile 
model.compile( optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

#-- Fit 
model.fit( feat, labels, epochs=50, batch_size=64,shuffle=True )
model.summary()


#-- Plot data 
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=p_feat[:,0], 
    y=p_feat[:,1], 
    mode='markers',  # Scatter plot with points
    marker=dict(size=6, color='green', opacity=0.7),
    name="Positive Samples"
))

fig.add_trace(go.Scatter(
    x=n_feat[:,0], 
    y=n_feat[:,1], 
    mode='markers',  # Scatter plot with points
    marker=dict(size=6, color='red', opacity=0.7),
    name="Negative Samples"
))

#
kernel = None
bias = None 
for var in model.trainable_variables:
    if 'kernel' in var.name:
        kernel = var.numpy()  # Access the kernel (weights) of the layer
    elif 'bias' in var.name:
        bias = var.numpy()  # Access the bias of the layer

a = kernel[0,0]
b = kernel[1,0] 
c = bias[0]
xx = np.linspace(-6, 6, 100)
yy = xx* -b/a + -c/a
fig.add_trace(go.Scatter(
    x=xx,
    y=yy,
    mode='lines',
    name=f'{b}x + {a}y + {c} = 0',
    line=dict(color='blue')
))

fig.show()




