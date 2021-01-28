%tensorflow_version 1.x
import tensorflow as tf
import numpy as np
print(tf.__version__)
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
plt.imshow(image_binarization(image_orig))

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

line_segmentation(image_binarization(image_orig))
def word_segmentation(img):
  start_matrix=[]
  end_matrix=[]
  dissection_matrix=[]

  vertical_hist = np.sum(img,axis=0,keepdims=True)/255
  start_count=0
  for i in range(len(vertical_hist)):
    if vertical_hist[i]>0 and vertical_hist[i-1]==0:
      start_count+=1
      start_matrix.append(i)
    if vertical_hist[i]==0 and start_count>0 and vertical_hist[i-1]>0:
      end_matrix.append(i)

  for i in range(len(start_matrix)):
    dissection_matrix.append([start_matrix[i],end_matrix[i]])
  return dissection_matrix
