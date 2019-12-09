import numpy as np
from scipy import signal
from scipy.signal import convolve2d as convolve
from collections import Counter
from matplotlib import pyplot as plt
# EXTRACTDIGITFEATURES extracts features from digit images
#   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
#   images X of the provided FEATURETYPE. The images are assumed to the of
#   size [W H 1 N] where the first two dimensions are the width and height.
#   The output is of size [D N] where D is the size of each feature and N
#   is the number of images. 
def pixelFeatures(x):
    features = x.reshape(x.shape[0]*x.shape[1], -1)
    return features

"""
def computeHist(mag, ori, bins,binSize, numOr):
    features = np.zeros(numOr)
    for i in range(binSize):
        for j in range(binSize):
            diff = ori[i][j].reshape(1,-1) - bins[:,np.newaxis]
            #print(diff[:,0])
            print(diff.shape)
            idx = np.where(diff == np.min(diff, axis = 0, keepdims=True))[0]
            print(idx.shape)
            idx1 = []
            for k in range(idx.shape[0]):
                if idx[k] == 0:
                    idx1.append(7)
                elif idx[k] == 7:
                    idx1.append(0)
                else:
                    if (np.abs(ori[i][j][k] - bins[idx[k]-1]) 
            < np.abs(ori[i][j][k] - bins[idx[k]+1])):
                        idx1.append(idx[k]-1)
                    else:
                        idx1.append(idx[k]+1)
            #idx1 = [idx-1 if (np.abs(ori[i][j][k] - bins[idx[k]-1]) 
            #< np.abs(ori[i][j][k] - bins[idx[k]+1])) else idx+1 for k in range(idx.shape[0])]
            print(len(idx1))

            ori[i][j].reshape(1,-1) - bins[idx,np.newaxis]
"""            
def computeHist(mag, ori, bins,binSize, numOr):
    features = np.zeros((numOr,mag.shape[2]))
    binspace = bins[1]-bins[0]
    for i in range(binSize):
        for j in range(binSize):
            diff = ori[i][j].reshape(1,-1) - bins[:,np.newaxis]
            sorted_id = np.argsort(np.abs(diff), axis=0)

            idx = sorted_id[0,:]
            idx1 = sorted_id[1,:]  
            #idx1[np.where(idx == 0)[0]] = 7
            #idx1[np.where(idx == 7)[0]] = 0
            
            val1 = np.abs(ori[i][j] - bins[idx])
            #val2 = np.abs(ori[i][j] - bins[idx1])
            temp = np.where(ori[i][j] > bins[7])[0]
            idx1[temp] = 0
            
                
            features[idx1,:] += val1*mag[i][j]/binspace
            features[idx,:] += mag[i][j]*(1-(val1/binspace))
    return features
            
def hogFeatures(x):
    numOr = 8
    binSize = 4
    #x = np.sqrt(x)
    gx_filt = np.asarray([-1,0,1]).reshape(1,-1)
    gy_filt = np.asarray([[-1],[0],[1]]).reshape(-1,1)
    Gx = np.zeros(x.shape)
    Gy = np.zeros(x.shape)
    for i in range(x.shape[2]):
        Gx[:,:,i] = convolve(x[:,:,i], gx_filt, mode='same')
        Gy[:,:,i] = convolve(x[:,:,i], gy_filt, mode='same')
    #print('shape',Gx.shape, Gy.shape)
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    #print(gradient_magnitude.shape)
    #print(np.max(Gx), np.max(Gy))
    gradient_orientation = np.arctan(Gy/(Gx+0.00000001))*180/np.pi + 90
    #print(gradient_orientation)
    #print(np.min(gradient_orientation), np.max(gradient_orientation))
    
    bins = np.linspace(0.,180.,numOr, endpoint=False)
    #print(bins)
    features = np.zeros((1,x.shape[2]))
    for i in range(0,x.shape[0],binSize):
        for j in range(0,x.shape[1],binSize):
            magBin = gradient_magnitude[i:i+binSize, j:j+binSize,:]
            oriBin = gradient_orientation[i:i+binSize, j:j+binSize,:]
            #print(magBin.shape, oriBin.shape)
            out = computeHist(magBin, oriBin, bins, binSize, numOr)
            features = np.concatenate((features, out), axis = 0)
            #print(features.shape)
    features = features[1:,:]
    
    #return np.sqrt(features/np.sqrt(np.sum(features, axis = 0, keepdims=True)))
    return features

def lbpFeatures(x):
    features = np.zeros((256, x.shape[2]))
    bitNum = np.asarray([[1,2,4],[8,0,16],[32,64,128]])
    for ind in range(x.shape[2]):
        singleImage = x[:,:,ind]
        listInt = []
        c = Counter()
        for i in range(1,x.shape[0]-1):
            for j in range(1,x.shape[1]-1):
                patch = singleImage[i-1:i+2,j-1:j+2] - singleImage[i,j]
                patch[patch>0] = 1
                patch[patch<0] = 0
                patch[1,1] = 0
                mappedInt = np.sum(patch*bitNum)
                listInt.append(mappedInt)
        c.update(listInt)
        indices = map(int, c.keys())
        c[0] = 0
        features[indices,ind] = c.values()
        #print(c.values()[0])
        #/np.linalg.norm(c.values())
        #output = np.sqrt(features)
        #features[:,ind] = features[:,ind]/np.sqrt(np.sum(features[:,ind]))
    #features = np.sqrt(features)
    #features  = features/np.sqrt(np.sum(features, axis=0, keepdims=True))
    return features
                
def extractDigitFeatures(x, featureType):
    
    if featureType == 'pixel':
        features = pixelFeatures(x)  # implement this
    elif featureType == 'hog':
        features = hogFeatures(x)  # implement this
    elif featureType == 'lbp':
        features = lbpFeatures(x)  # implement this        

    return features
