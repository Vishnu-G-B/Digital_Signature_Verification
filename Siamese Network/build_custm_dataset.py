import os, shutil,cv2
import numpy as np


index = 0
label = 1 
labels = np.zeros(887,dtype=int)
dir_path = r"C:\Users\vishn\OneDrive\Desktop\python projects\Handrecog_proj\Siamese Network\archive\sign_data\sign_data\train"
images = []

for folder_name in os.listdir(dir_path):

    if folder_name.isdigit() and (int(folder_name)>=1 and int(folder_name)<=69):
        folder_path = os.path.join(dir_path,folder_name)

        for file_name in os.listdir(folder_path):

            if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png') or file_name.endswith('.PNG'):

                img = cv2.imread(os.path.join(folder_path,file_name))
                img = cv2.resize(img,(224,224))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # img = img/255
                if img is not None:
                    images.append(img)
                # src_path = os.path.join(folder_path,file_name)
                # dst_path = os.path.join(r"C:\Users\vishn\OneDrive\Desktop\python projects\Handrecog_proj\Siamese Network\Signature_dataset\TRAINING IMAGES", file_name)
                # shutil.copy(src_path,dst_path)
                labels[index] = label
                index+=1
        label+=1

# np.save("Handrecog_proj/Siamese Network/Signature_dataset/LABELS", labels)

img_array = np.array(images)


np.save("Handrecog_proj\Siamese Network\Signature_dataset\image_nparr",img_array)

# my_arr = np.load("Signature_dataset/image_nparr.npy")

# for i in range(len(my_arr)):
#     cv2.imshow("test",my_arr[i])
#     cv2.waitKey(300)

# cv2.destroyAllWindows()
