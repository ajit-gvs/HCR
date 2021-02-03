
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_orig=Image.open('iam.png','r')

def image_binarization(image):
  image_grayscale=image.convert('L')
  img=np.array(image_grayscale)
  img[img <128] = 0# white
  img[img >=128] = 254 # black
  img[img==0]=255
  img[img==254]=0
  
  return img


def line_segmentation(img):
  start_matrix=[]
  end_matrix=[]
  dissection_matrix=[]

  horizontal_hist = np.sum(img,axis=1,keepdims=True)/255
  start_count=0
  for i in range(len(horizontal_hist)):
    if horizontal_hist[i]>0 and horizontal_hist[i-1]==0:
      start_count+=1
      start_matrix.append(i)
    if horizontal_hist[i]==0 and start_count>0 and horizontal_hist[i-1]>0:
      end_matrix.append(i)

  for i in range(len(start_matrix)):
    dissection_matrix.append([start_matrix[i],end_matrix[i]])
  return dissection_matrix

def word_segmentation(img):
  start_matrix=[]
  end_matrix=[]

  #matrix to get the start and end points of a word
  dissection_matrix=[]
  
  length=[]
  vertical_hist = np.sum(img,axis=0,keepdims=True)/255
  
  
  start_count=0
  print(len(vertical_hist))
  
  for i in range(len(vertical_hist[0])):
    if vertical_hist[0][i]>0 and vertical_hist[0][i-1]==0:
      start_count+=1
      start_matrix.append(i)
    if vertical_hist[0][i]==0 and start_count>0 and vertical_hist[0][i-1]>0:
      end_matrix.append(i)
  
  
  length_mag=0
  for i in range(len(start_matrix)):
    if i>0:
      length_mag=(start_matrix[i]-end_matrix[i-1])
      length.append(length_mag)

  max=np.max(length)
  min=np.min(length)    
  avg=max/3

  dissection_matrix.append([start_matrix[0],end_matrix[0]])
  j=0
  for i in range(len(length)):
    
    if length[i]> avg:
      dissection_matrix.append([start_matrix[i+1],end_matrix[i+1]])
      j+=1
      
    if length[i]<=avg:
      dissection_matrix[j][1]=end_matrix[i+1]
      
  print(start_matrix)
  print(length)
  print(end_matrix)
  print(dissection_matrix)

  return dissection_matrix
