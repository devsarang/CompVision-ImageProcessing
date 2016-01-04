import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import sys
import cv2

np.set_printoptions(threshold='nan')
sys.setrecursionlimit(15000)

threshold1 = 10
threshold2 = 10

finalMerged = []
neighborList = {}
meanIntList = {}
def regionGrowing(x,y,label,meanInt):
    if x<0 or x>img.shape[0]-1 or y<0 or y>img.shape[1]-1 or labelR[x][y] != 0 or abs(img[x][y] - meanInt) > threshold1:
        return
    labelR[x][y] = label
    #meanInt = (pixLabeled*meanInt + img[x][y])/(pixLabeled+1)
    regionGrowing(x,y+1,label,img[x][y])
    regionGrowing(x+1,y,label,img[x][y])
    regionGrowing(x,y-1,label,img[x][y])
    regionGrowing(x-1,y,label,img[x][y])
    regionGrowing(x-1,y+1,label,img[x][y])  
    regionGrowing(x+1,y+1,label,img[x][y]) 
    regionGrowing(x+1,y-1,label,img[x][y])
    regionGrowing(x-1,y-1,label,img[x][y])
        
def findNeighbors(label):
    retList = []
    index = np.where(labelR == label)
    for x, y in zip(index[0], index[1]):
            if labelR[x][y] == label and x!=0 and x!=img.shape[0]-1 and y!=0 and y!=img.shape[1]-1:
                if labelR[x][y+1]!=label:
                    if labelR[x][y+1] not in retList:
                        retList.append(labelR[x][y+1])
                if labelR[x+1][y]!=label:
                    if labelR[x+1][y] not in retList:
                        retList.append(labelR[x+1][y])
                if labelR[x][y-1]!=label:
                    if labelR[x][y-1] not in retList:
                        retList.append(labelR[x][y-1])
                if labelR[x-1][y]!=label:
                    if labelR[x-1][y] not in retList:
                        retList.append(labelR[x-1][y])
    return retList

def findBoundary(label):
    retList=[]
    retList.append([])
    retList.append([])
    index = np.where(labelR == label)
    for x, y in zip(index[0], index[1]):
            if labelR[x][y] == label and x!=0 and x!=img.shape[0]-1 and y!=0 and y!=img.shape[1]-1:
                if labelR[x][y+1]!=label or labelR[x+1][y]!=label or labelR[x][y-1]!=label or labelR[x-1][y]!=label:
                    retList[0].append(x)
                    retList[1].append(y)
    return retList

def calculateMean(label):
    sumInt = 0
    count = 0
    index = np.where(labelR == label)
    for x, y in zip(index[0], index[1]):
        if labelR[x][y] == label:
            sumInt = sumInt + img[x][y]
            count = count + 1
    return sumInt/count
                     
                     
def merge(label,neighbor):
    index = np.where(labelR == neighbor)
    for x, y in zip(index[0], index[1]):
        labelR[x][y] = label
  
def recursiveMerge(original_label,label):
    if label in finalMerged:
        return
    for neighbor in neighborList[label]:
        if abs(meanIntList[label]-meanIntList[neighbor])<threshold2:
            finalMerged.append(label)
            merge(original_label,neighbor)
            recursiveMerge(original_label,neighbor)
            
# Load the image
img = imread('Peppers.jpg')
label = 0
labelR = np.zeros(img.shape,np.int64)
for x in range(0,img.shape[0]):
    for y in range(0,img.shape[1]):
         if labelR[x][y] == 0:
             print label
             label = label+1
             regionGrowing(x,y,label,img[x][y])
for labels in range(1,labelR.max()+1): 
    print "mean and neighbors"
    print labels
    meanIntList[labels]=calculateMean(labels)
    neighborList[labels] = findNeighbors(labels)


i=1
listSize = labelR.max()
print "list size"
print listSize
while i <= listSize:
    print "recursive merge"
    print i
    if i not in finalMerged:
        recursiveMerge(i,i)
    i=i+1

labelR=labelR%255

plt.figure(1)  
plt.imshow(labelR, cmap = 'gray',interpolation='none')
plt.show()

#plt.figure(2)  
#plt.imshow(img, cmap = 'gray',interpolation='none')
#for labels in range(1,listSize+1): 
#    index = findBoundary(labels)
#    print index
#    for x, y in zip(index[0], index[1]):
#        if len(index[0]) > 100:
#            img[x][y] = 255
#plt.show()
            