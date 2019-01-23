import os
import numpy as np
from scipy import misc
from skimage.transform import resize

def imgFeatures(img, i, photos):
    features = np.zeros((len(img),3+len(photos)))
    features[:,0:3] = img
    for i in range(len(img)):
        c = img[i]
        for p, oimg in enumerate(photos):
            features[i,3+p] = -np.sqrt( np.min( (c[0]-oimg[:,0])*(c[0]-oimg[:,0]) + (c[1]-oimg[:,1])*(c[1]-oimg[:,1]) + (c[2]-oimg[:,2])*(c[2]-oimg[:,2]) ) )
    return features

def rangeWithout(without, to):
    r = list(range(to))
    del r[without]
    print(r)
    return r

def removeDuplicites(data, duplicityAccuracu):
    roundedData = np.round(data[:,0:3], duplicityAccuracu)
    sortedIndex = np.lexsort(roundedData.T)
    sortedData =  roundedData[sortedIndex,:]
    rowMask = np.append([True],np.any(np.diff(sortedData,axis=0),1))
    return sortedData[rowMask]

def extractData(path, dataLimit, removeDupliciteColors, duplicityAccuracu):

    print("EXTRACTING DATA..")

    data = None

    photos = os.listdir(path)
    size = int(round(np.sqrt(dataLimit / (len(photos)))))

    print("Loading", len(photos), "photos..")
    print("Using size", (size, size))

    photos = [resize(misc.imread(path+photo), (size, size), anti_aliasing=False) for photo in photos]
    photos = [img.reshape(-1, img.shape[-1]) for img in photos]

    print("Extracting relations..")

    i = 0
    c = 50
    for img in photos:
        if data is None:
            data = imgFeatures(img, i, photos)
        else:
            data = np.concatenate((data, imgFeatures(img, i, photos)))

        if data.shape[0] % c == 0:
            print(str(int(round(data.shape[0] / dataLimit * 100)))+"%")
            c += 1
        if data.shape[0] > dataLimit:
            break

        i += 1

    if removeDupliciteColors:
        print("Removing duplicite colors..")
        data = removeDuplicites(data, duplicityAccuracu)

    print("Final data size is", data.shape, ", saving..")

    np.save("output/data.npy", data)

    return data
