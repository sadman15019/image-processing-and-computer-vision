# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 02:06:55 2023

@author: Asus
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt



def toeplitz(x,y):
    arr=np.zeros((len(x),len(y)),dtype=np.int32)
    start=0 #start is used to know where non zero value starts for each column
    for i in range(arr.shape[1]):#col iterate
        for j in range(arr.shape[0]):#row iterate
            arr[j,i]=x[j-start]
        start+=1
    return arr

def main():
    img=cv2.imread("rsz_lena.png",cv2.IMREAD_GRAYSCALE)

    kernel= np.array(([[0,1,0],
                      [1,-4,1],
                      [0,1,0]]),np.float32)#laplace filter

    img_h=img.shape[0]
    img_w=img.shape[1]
    k_h=kernel.shape[0]
    k_w=kernel.shape[1]

    out_h=img_h+k_h-1
    out_w=img_w+k_w-1

    x=k_h//2
    y=k_w//2

    #img=cv2.copyMakeBorder(img,top=x,bottom=x,left=y,right=y,borderType= cv2.BORDER_CONSTANT)
    fil=cv2.copyMakeBorder(kernel,top=out_h-k_h,bottom=0,left=0,right=out_w-k_w,borderType= cv2.BORDER_CONSTANT)#only up and right padding 

        
    #create toeplitz matrix
    tplz_list=[]
    for i in range(fil.shape[0]-1,-1,-1):
        x=fil[i,:]#create toepletz matrix for each of row of filter
        y=np.r_[x[0], np.zeros(img_w-1)] #number of column same as number of column of input image,creating first row
        tplz=toeplitz(x,y)
        tplz_list.append(tplz)
    
    #create indices for doubly blocked toeplitz matrix
    x = range(1, fil.shape[0]+1)#number of rows same as number or rows in filter
    y = np.r_[x[0], np.zeros(img_h-1, dtype=int)]#number of column same as number of rows of input image,creating first row
    doubly_indices = toeplitz(x, y)

    teoplitz_shape=tplz_list[0].shape
    x=teoplitz_shape[0]*doubly_indices.shape[0]#number of rows in each single toeplitz matrix mul with number of rows for double blocked matrix
    y=teoplitz_shape[1]*doubly_indices.shape[1]#number of col in each single toeplitz matrix mul with number of col for double blocked matrix
    doubly_blocked=np.zeros((x,y))


    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i=i*teoplitz_shape[0]#base index for corresponding toeplitz matrix
            start_j=j*teoplitz_shape[1]
            end_i=start_i+teoplitz_shape[0]
            end_j=start_j+teoplitz_shape[1]
            doubly_blocked[start_i:end_i,start_j:end_j]=tplz_list[doubly_indices[i,j]-1]#in doubly_indices zero indexing,in doubly blocked 1 indexing
            
    vector_mat=np.zeros(img_h*img_w,dtype=img.dtype)
    k=0
    for i in range (img.shape[0]-1,-1,-1):
        for j in range(0,img.shape[1]):
            vector_mat[k]=img[i,j]#image er ulta dik theke vector er shurur dik theke
            k+=1

    result_vector = np.matmul(doubly_blocked, vector_mat)  
    output_image = np.zeros((out_h,out_w),dtype = np.float32)
    j=img.shape[0]-1
    for i in range(out_h):
           start = i * out_h
           end = start + out_w
           output_image[j,:] = result_vector[start:end] #image er ulta dik theke,result er shurur dik theke
           j-=1
    output_image/=255


    cv2.imshow("input",img)
    cv2.imshow("image",output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    


    plt.imshow(output_image,'gray')
    plt.title("Output for laplacian " )
    plt.show()
if __name__=="__main__":
    main()
        