#import fileinput import filename
from typing import Concatenate
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read(filename):
    img=cv2.imread(filename)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    #plt.show()
    return img

filename= "ghat.jpg"
img= read(filename)

#create edge mask

def edge_mask(img, line_size, blur_value):
    """
    input: Input image
    output: Gray scale image
    """
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur= cv2.medianBlur(gray, blur_value)

    edges= cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, line_size, blur_value)
    return edges
    
line_size, blur_value=7,7
edges=edge_mask(img, line_size, blur_value)
plt.imshow(edges, cmap="gray")
#plt.show()

#reducing color pallette

def color_quant(img, k):
    #k here defines how many colors we want to be highlighted
    #transform img
    data=np.float32(img).reshape((-1,3))

    #determine criteria
    criteria= (cv2.TermCriteria_EPS+ cv2.TERM_CRITERIA_MAX_ITER,20,0.001)

    #implimenting K-means

    ret, label, center = cv2.kmeans(data,k,None,criteria,10, cv2.KMEANS_RANDOM_CENTERS)
    center= np.uint8(center)

    result= center[label.flatten()]
    result= result.reshape(img.shape)

    return result

img = color_quant(img, k=9)
plt.imshow(img)
#plt.show()

#reduce noise

blurred= cv2.bilateralFilter(img, d=3, sigmaColor=200, sigmaSpace=200)
plt.imshow(blurred)
#plt.show()
#combine edge mask with quantize img

def cartoon():
    c= cv2.bitwise_and(blurred, blurred, mask=edges)

    plt.imshow(c)
    plt.title("cartoonified img")
    #plt.show()
    plt.imshow(img)
    #plt.title("org_img")
    #plt.show()
    combined=np.concatenate((c,img),axis=0)
    plt.imshow(combined)
    plt.show()
cartoon()
