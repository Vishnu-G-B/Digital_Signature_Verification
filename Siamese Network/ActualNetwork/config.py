import os

img_shape  = (224,224,1)

batch_size = 64
epochs = 10

base_path = "Output"
model_path = os.path.sep.join([base_path, "siamese_model"])
plot_path = os.path.sep.join([base_path, "plot.png"])
