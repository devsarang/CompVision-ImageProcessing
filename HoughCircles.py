import numpy as np
import cv2
import datetime

np.set_printoptions(threshold='nan')

def gaussianBlur(img):
    #gaussian blur by a 3X3 kernel
    return cv2.GaussianBlur(img, (3,3),0)
    
def cannyEdge(img):
    #edge detection and thresholding by hysterisys 80 is the lower threshold and 160 is the upper one 
    return cv2.Canny(gaussianImg,80,160)
    
def houghTransformAccumulatorcell(img,r1,r2,minDist):
    start_time = datetime.datetime.now()
    localMaxima = []
    radiusMin = r1-3
    radiusMax = r2+3
    threshold = 3*radiusMin            #approximately half the circumference points of min radius should meet at a point  
    radDiff = radiusMax - radiusMin
    accumulatorCell=np.zeros((radDiff,img.shape[0],img.shape[1]),np.int8)
    
    index = np.where(img == 1)
    for r in range(0,radDiff):
        for x, y in zip(index[0], index[1]):
            #using the mid point circle drawing algorithm for drawing circle in parameter space
            a=0
            b=r+radiusMin
            D=3-2*(r+radiusMin) 
            while(a<b):
                if x+a>=0 and x+a<img.shape[0] and y+b>=0 and y+b<img.shape[1]:
                    accumulatorCell[r][x+a][y+b] += 1
                if x-a>=0 and x-a<img.shape[0] and y+b>=0 and y+b<img.shape[1]:
                    accumulatorCell[r][x-a][y+b] += 1
                if x+a>=0 and x+a<img.shape[0] and y-b>=0 and y-b<img.shape[1]:
                    accumulatorCell[r][x+a][y-b] += 1
                if x-a>=0 and x-a<img.shape[0] and y-b>=0 and y-b<img.shape[1]:
                    accumulatorCell[r][x-a][y-b] += 1
                if x+b>=0 and x+b<img.shape[0] and y+a>=0 and y+a<img.shape[1]:
                    accumulatorCell[r][x+b][y+a] += 1
                if x-b>=0 and x-b<img.shape[0] and y+a>=0 and y+a<img.shape[1]:
                    accumulatorCell[r][x-b][y+a] += 1
                if x+b>=0 and x+b<img.shape[0] and y-a>=0 and y-a<img.shape[1]:
                    accumulatorCell[r][x+b][y-a] += 1
                if x-b>=0 and x-b<img.shape[0] and y-a>=0 and y-a<img.shape[1]:
                    accumulatorCell[r][x-b][y-a] += 1
                a += 1
                if D<0:
                    D=D+4*a+6
                else:
                    D=D+4*(a-b)+10
                    b -= 1   
    
    #uncomment it to see the intermediate slices of accumulator cell for each radius
    #for r in range(0,radDiff):
    #    cv2.imshow('Test'+str(r),accumulatorCell[r])
    
    #threshold for minimum number of cicles passing throug a cell to be considered for local maxima
    points = np.where(accumulatorCell > threshold)
    
    #getting the local maxima by computing the maximum point in a region of 6x10x10 of the accumulator array 
    for r,x,y in zip(points[0], points[1],points[2]):
        if r-3>0 and r+4<radDiff and x-5>0 and x+6<img.shape[0] and y-5>0 and y+6<img.shape[1]:
            if accumulatorCell[r-3:r+4,x-5:x+6,y-5:y+6].max() == accumulatorCell[r][x][y]:
                localMaxima.append((r+radiusMin,x,y)) 
    
    #remove the points which are too close to each other
    retList = removeCloseCenter(localMaxima,minDist) 
    end_time = datetime.datetime.now()
    print "Accumulator Cell Based Hough Transform Time Taken:"  
    print end_time - start_time                  
    return retList      
    
def houghTransform4Points(img,r1,r2,minDist):
    start_time = datetime.datetime.now()
    localMaxima = []
    threshold1 = 2       #only points with minimum of three intersection is considered for center
    threshold2 = 6       #only points with minimum sum of 7 in 2X2 window is considered for center
    radiusMin = r1
    radiusMax = r2
    (contours,heirarchy)= cv2.findContours(img, cv2.cv.CV_RETR_LIST, cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contourLen = len(contour)
        p = []
        #only consider figures with more pixels than the the half of the circumference of the min radius circle for checking circles
        if(contourLen>3*radiusMin):
            p.append(contour[0]) 
            p.append(contour[round(contourLen/4)])
            p.append(contour[round(contourLen/2)])
            p.append(contour[round(3*contourLen/4)])
            for r in range(radiusMin,radiusMax):
                imgArr = np.zeros((img.shape))
                for i in range(len(p)):
                    x=p[i][0][1]
                    y=p[i][0][0]
                     #using the mid point circle drawing algorithm for drawing circle 2D array
                    a=0
                    b=r
                    D=3-2*(r+radiusMin) 
                    while(a<b):
                        if x+a>=0 and x+a<img.shape[0] and y+b>=0 and y+b<img.shape[1]:
                            imgArr[x+a][y+b] += 1
                        if x-a>=0 and x-a<img.shape[0] and y+b>=0 and y+b<img.shape[1]:
                            imgArr[x-a][y+b] += 1
                        if x+a>=0 and x+a<img.shape[0] and y-b>=0 and y-b<img.shape[1]:
                            imgArr[x+a][y-b] += 1
                        if x-a>=0 and x-a<img.shape[0] and y-b>=0 and y-b<img.shape[1]:
                            imgArr[x-a][y-b] += 1
                        if x+b>=0 and x+b<img.shape[0] and y+a>=0 and y+a<img.shape[1]:
                            imgArr[x+b][y+a] += 1
                        if x-b>=0 and x-b<img.shape[0] and y+a>=0 and y+a<img.shape[1]:
                            imgArr[x-b][y+a] += 1
                        if x+b>=0 and x+b<img.shape[0] and y-a>=0 and y-a<img.shape[1]:
                            imgArr[x+b][y-a] += 1
                        if x-b>=0 and x-b<img.shape[0] and y-a>=0 and y-a<img.shape[1]:
                            imgArr[x-b][y-a] += 1
                        a += 1
                        if D<0:
                            D=D+4*a+6
                        else:
                            D=D+4*(a-b)+10
                            b -= 1   
                #only points with minimum of three intersection is considered for center
                points = np.where(imgArr > threshold1)
                for x,y in zip(points[0],points[1]):  
                    #only points with minimum sum of 7 in 2X2 window is considered for center          
                    if(imgArr[x-1:x+1,y-1:y+1].sum() > threshold2):
                        localMaxima.append((r,x,y))
                        
    #remove the points which are too close to each other
    retList = removeCloseCenter(localMaxima,minDist)    
    end_time = datetime.datetime.now()
    print "4 Points Based Hough Transform Time Taken:"  
    print end_time - start_time    
    return retList           
                    
#removes points in a list with values closer than the minDist
def removeCloseCenter(maximaList,minDist):
    length = len(maximaList)
    i=0
    while i < length:
        j = i+1 
        while j < length:
            if i<length and j<length:
                if((np.sqrt((maximaList[i][1]-maximaList[j][1])**2 + (maximaList[i][2]-maximaList[j][2])**2) < minDist)):
                    maximaList.pop(j)
                    j-=1   
                    length-=1
            j+=1
        i+=1
    return maximaList
    
    
# Load the image
img = cv2.imread('HoughCircles.jpg',1) 
imgCopy1 = img.copy()
imgCopy2 = img.copy()
                          
gaussianImg=gaussianBlur(img)

cv2.imshow('Gussian Blur Image',gaussianImg)

edgeImg = cannyEdge(gaussianImg)
binImg = edgeImg/255                #converting into a binary image

cv2.imshow('Canny Edge Image',edgeImg)

#accumulator based Hough transform function with parameters Edge detected binary image,min radius,max radius, min distance between two centers
points = houghTransformAccumulatorcell(binImg,30,55,15)

for i in points:
    # draw the outer circle
    cv2.circle(img,(i[2],i[1]), i[0],(255,0,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[2],i[1]),1,(0,255,0),3)
    
cv2.imshow('Hough Transform Circle',img)

points = houghTransform4Points(binImg,25,55,15)
for i in points:
    # draw the outer circle
    cv2.circle(imgCopy1,(i[2],i[1]), i[0],(0,255,255),2)
    # draw the center of the circle
    cv2.circle(imgCopy1,(i[2],i[1]),1,(255,0,0),3)
   
cv2.imshow('Hough Transform 4 Points Circle',imgCopy1)

circles = cv2.HoughCircles(edgeImg,cv2.cv.CV_HOUGH_GRADIENT,1,40,
              param1=160,
              param2=15,
              minRadius=30,
              maxRadius=55)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(imgCopy2,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(imgCopy2,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('CV2 Hough Transform Circle',imgCopy2)
cv2.waitKey(0)
cv2.destroyAllWindows()


