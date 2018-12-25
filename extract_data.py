import os
import numpy as np
from scipy import misc
from skimage.transform import resize

PHOTOS_PATH = '/home/ienze/Pictures/tmp/'
COUNT_LIMIT = 9000
SIZE = 5
# PHOTOS_PATH = '/run/media/ienze/D/Photo/sierpinsky/'

data = None

def imgFeatures(img, i, photos):
    features = np.zeros((len(img),3+len(photos)))
    features[:,0:3] = img
    for i in range(len(img)):
        c = img[i]
        for p, oimg in enumerate(photos):
            closestDistance = 6.0
            for j in range(len(oimg)):
                oc = oimg[j]
                distance = np.sqrt( (c[0]-oc[0])*(c[0]-oc[0]) + (c[1]-oc[1])*(c[1]-oc[1]) + (c[2]-oc[2])*(c[2]-oc[2]) )
                if distance < closestDistance:
                    closestDistance = distance
            features[i,3+p] = -1 * closestDistance
    return features

photos = os.listdir(PHOTOS_PATH)
SIZE = int(round(np.sqrt(COUNT_LIMIT / (len(photos)))))
print("Loading", len(photos), "photos..")
print("Using size", SIZE)
photos = [resize(misc.imread(PHOTOS_PATH+photo), (SIZE, SIZE), anti_aliasing=False) for photo in photos]
photos = [img.reshape(-1, img.shape[-1]) for img in photos]
print("Extracting data..")
i = 0
for img in photos:
    if data is None:
        data = imgFeatures(img, i, photos)
    else:
        data = np.concatenate((data, imgFeatures(img, i, photos)))

    if data.shape[0] % 100 == 0:
        print(str(int(round(data.shape[0] / COUNT_LIMIT * 100)))+"%")
    if data.shape[0] > COUNT_LIMIT:
        break

    i += 1

print("Final data size is", data.shape, ", saving..")

np.save("data.npy", data)
