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
    start=0
    for i in range(arr.shape[1]):
        for j in range(arr.shape[0]):
            arr[j,i]=x[j-start]
        start+=1
    return arr

def main():
    img=cv2.imread("rsz_lena.png",cv2.IMREAD_GRAYSCALE)

    kernel= np.array(([[0,1,0],
                      [1,-4,1],
                      [0,1,0]]),np.float32)

    img_h=img.shape[0]
    img_w=img.shape[1]
    k_h=kernel.shape[0]
    k_w=kernel.shape[1]

    out_h=img_h+k_h-1
    out_w=img_w+k_w-1

    x=k_h//2
    y=k_w//2

    #img=cv2.copyMakeBorder(img,top=x,bottom=x,left=y,right=y,borderType= cv2.BORDER_CONSTANT)
    fil=cv2.copyMakeBorder(kernel,top=out_h-k_h,bottom=0,left=0,right=out_w-k_w,borderType= cv2.BORDER_CONSTANT)

        
    #create toeplitz matrix
    tplz_list=[]
    for i in range(fil.shape[0]-1,-1,-1):
        x=fil[i,:]
        y=np.r_[x[0], np.zeros(img_w-1)] 
        tplz=toeplitz(x,y)
        tplz_list.append(tplz)
        
    x = range(1, fil.shape[0]+1)
    y = np.r_[x[0], np.zeros(img_w-1, dtype=int)]
    doubly_indices = toeplitz(x, y)

    teoplitz_shape=tplz_list[0].shape
    x=teoplitz_shape[0]*doubly_indices.shape[0]
    y=teoplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked=np.zeros((x,y))


    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i=i*teoplitz_shape[0]
            start_j=j*teoplitz_shape[1]
            end_i=start_i+teoplitz_shape[0]
            end_j=start_j+teoplitz_shape[1]
            doubly_blocked[start_i:end_i,start_j:end_j]=tplz_list[doubly_indices[i,j]-1]
            
    vector_mat=np.zeros(img_h*img_w,dtype=img.dtype)
    k=0
    for i in range (img.shape[0]-1,-1,-1):
        for j in range(0,img.shape[1]):
            vector_mat[k]=img[i,j]
            k+=1

    result_vector = np.matmul(doubly_blocked, vector_mat)  
    output_image = np.zeros((out_h,out_w),dtype = np.float32)
    for i in range(out_h):
           start = i * out_h
           end = start + out_w
           output_image[i,:] = result_vector[start:end] 
    output_image = np.flipud(output_image)
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
        