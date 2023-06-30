#@title Testing Script for Neural Net
from ActualNetwork import config, utils
from ActualNetwork.siamese_network import build_siamese_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import cv2, extract_testimgs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Lambda

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,help="path to input directory of testing images")
# args = vars(ap.parse_args())

# extract_testimgs.prep_test_imgs()

# grab the test dataset image paths and then randomly generate a
# total of 10 image pairs
print("[INFO] loading test dataset...")
testImagePaths = list(list_images("examples"))
np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(10, 2))

# load the model from disk
# print("[INFO] loading siamese model...")
# model = load_model(config.model_path)

# Rebuild Model from disk
print("[INFO] re-building siamese network...")
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
model.load_weights("Output/weights_kaggle.h5")


# loop over all image pairs
for (i, (pathA, pathB)) in enumerate(pairs):
    # load both the images and convert them to grayscale
    imageA = cv2.imread(pathA, 0)
    imageB = cv2.imread(pathB, 0)
    # create a copy of both the images for visualization purpose
    origA = imageA.copy()
    origB = imageB.copy()
    # add channel a dimension to both the images
    imageA = np.expand_dims(imageA, axis=-1)
    imageB = np.expand_dims(imageB, axis=-1)
    # add a batch dimension to both images
    imageA = np.expand_dims(imageA, axis=0)
    imageB = np.expand_dims(imageB, axis=0)
    # scale the pixel values to the range of [0, 1]
    imageA = imageA / 255.0
    imageB = imageB / 255.0
    # use our siamese model to make predictions on the image pair,
    # indicating whether or not the images belong to the same class
    preds = model.predict([imageA, imageB])
    proba = preds[0][0]

    # initialize the figure
    fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
    plt.suptitle("Similarity: {:.2f}".format(proba))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the plot
    plt.show()
