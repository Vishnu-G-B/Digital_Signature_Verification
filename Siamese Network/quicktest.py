import numpy as np
import cv2
import os
from PIL import Image

images_arr = np.load("Handrecog_proj\Siamese Network\Signature_dataset\image_nparr.npy")
labels_arr = np.load("Handrecog_proj\Siamese Network\Signature_dataset\labels.npy")
numClasses = len(np.unique(labels_arr))


# for i,img in enumerate(images_arr):
#     print(labels_arr[i],end="==")
#     print(i)
    
for img in images_arr:
    # img = cv2.resize(img, (224, 224))
    img = Image.fromarray(img)
    img.show()
