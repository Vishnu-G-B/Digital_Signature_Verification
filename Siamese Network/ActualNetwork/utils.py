from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def make_pairs(images, labels):

    pairImages =[]
    pairLabels = []
    idx = []

    numClasses = len(np.unique(labels))
    # idx = [np.where(labels == i)[0] for i in range(1, numClasses+1)]
    # print(type(idx))
    # for element in labels:
    #     print(element,end= " :: ")
    
    for i in range(1, numClasses+1):
        idx.append(np.where(labels == i)[0])
    for i in idx:
        print(i)
    # raise KeyError
    try:
        # for idxA, currentImage in enumerate(images):
        #     label = labels[idxA]
            
        
        for idxA in range(len(images)):
            #Grabs the current image associated with indxA
            #Then grabs the label associated with that image
            currentImage = images[idxA]
            # img = Image.fromarray(currentImage,"RGB")
            # img.show()
            label = labels[idxA]

            # randomly pick an image that belongs to the *same* class
            print(label, idxA,end=" == ")
            idxB = np.random.choice(idx[label-1])
            print(idxB)
            posImage = images[idxB-1]
            # img2 = Image.fromarray(posImage,"RGB")
            # img2.show()
            
            # prepare a positive pair and update the images and labels
            # lists, respectively
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])

            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            negIdx = np.where(labels != label)[0]
            # print(negIdx)
            negImage = images[np.random.choice(negIdx)]
            # prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
            # return a 2-tuple of our image pairs and labels
        
    except IndexError:
        return (np.array(pairImages), np.array(pairLabels))

    return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
    # unpack the vectors into separate lists

    #thefk is happening
    featsA, featsB = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,keepdims=True)
    # return the euclidean distance between the vectors

    # print(K.sqrt(K.maximum(sumSquared, K.epsilon())))

    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)