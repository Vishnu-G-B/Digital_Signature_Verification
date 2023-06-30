#@title Training Script for Neural Net
from ActualNetwork.siamese_network import build_siamese_model
from ActualNetwork import config,utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Lambda
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

print("[INFO] loading signature dataset...")
# (trainX, trainY), (testX, testY) = mnist.load_data()
# trainX = trainX / 255.0
# testX = testX / 255.0
X = np.load("Handrecog_proj\Siamese Network\Signature_dataset\image_nparr.npy")
y = np.load("Handrecog_proj\Siamese Network\Signature_dataset\labels.npy")
X = X /255

# add a channel dimension to the images
X = np.expand_dims(X, axis=-1)
# testX = np.expand_dims(testX, axis=-1)

trainX, testX,trainY, testY = train_test_split(X,y,random_state=0)

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(X, y)
(pairTest, labelTest) = utils.make_pairs(testX, testY)

print("[INFO] building siamese network...")
imgA = Input(shape=config.img_shape)
imgB = Input(shape=config.img_shape)
featureExtractor = build_siamese_model(config.img_shape)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
# distance = Lambda(utils.euclidean_distance)([featsA, featsB])

distance_layer = Lambda(utils.euclidean_distance)
distance = distance_layer([featsA,featsB])

outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

#compiling the model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

# train the model
print("[INFO] training model...")
history = model.fit(
    [pairTrain[:, 0],pairTrain[:, 1]],labelTrain[:],
    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
    batch_size=config.batch_size,
    epochs=config.epochs)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.model_path)
model.save_weights("Output/weights_local.h5")

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.plot_path)