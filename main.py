# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 08:41:50 2022

@author: konan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

I=cv2.imread("boo.png",0)
plt.imshow(I)
plt.title("Image d'origine")
plt.show()

def upsample(img, scale):
     h, w = img.shape[:2]
     img = cv2.resize(img, (w*scale, h*scale))
     return img

def labelisation(h,w,x,y,labels,I):
    voisins = []
    if(x>0):
        if((labels[x-1,y] == 0)&(I[i][j]==I[x-1][y])): 
            voisins.append([x-1,y])
    if(x<h-1):
        if((labels[x+1,y] == 0)&(I[i][j]==I[x+1][y])):
            voisins.append([x+1,y])
    if(y>0):
        if((labels[x,y-1] == 0)&(I[i][j]==I[x][y-1])):
            voisins.append([x,y-1])
    if(y<w-1):
        if((labels[x,y+1] == 0)&(I[i][j]==I[x][y+1])):
            voisins.append([x,y+1])
    if((x>0)&(y>0)):
       if((labels[x-1,y-1] == 0)&(I[i][j]==I[x-1][y-1])): 
           voisins.append([x-1,y-1])
    if((x>0)&(y<w-1)):
       if((labels[x-1,y+1] == 0)&(I[i][j]==I[x-1][y+1])): 
           voisins.append([x-1,y+1])
    if((x<h-1)&(y>0)):
       if((labels[x+1,y-1] == 0)&(I[i][j]==I[x+1][y-1])): 
           voisins.append([x+1,y-1])
    if((x<h-1)&(y<w-1)):
       if((labels[x+1,y+1] == 0)&(I[i][j]==I[x+1][y+1])): 
           voisins.append([x+1,y+1])
    return voisins



#Labelisation

(h,w) = np.shape(I)
labels = np.zeros([h,w])
label = 0

for i in range(h):
    for j in range(w):
        if(labels[i,j]==0):
            label = label + 1
            labels[i,j] = label
            fin = True
            voisins = labelisation(h,w,i,j,labels,I)
            for k in range(len(voisins)):
                labels[voisins[k][0],voisins[k][1]] = labels[i,j]
            V = []
            while(fin):
                if(len(voisins)==0):
                    fin = False
                else:
                    V = labelisation(h,w,voisins[0][0],voisins[0][1],labels,I)
                    dernier_voisin = voisins.pop(0)
                    for l in range(len(V)):
                        voisins.append(V[l])
                        labels[V[l][0],V[l][1]] = labels[dernier_voisin[0],dernier_voisin[1]]
                        
plt.imshow(labels)
plt.title("Image de labels")
plt.show()
        

# x = 3
# y = 3
# test = np.ones([x,x])
# nx = int(h/x)
# ny = int(w/y)

# labels = []
# H_x = []
# H_y = []
# for i in range(nx):
#     for j in range(ny):
#         bloc = I[i*x:(i+1)*x,j*y:(j+1)*y]
#         milieu = bloc[int(x/2),int(y/2)]
#         if(int((np.sum(bloc))/(x*y)) == milieu):
#             grid = cv2.circle(I,(j*y+int(y/2),x*i+int(x/2)),radius=0, color = (int(255-milieu),0,0) , thickness = 1)
#             labels.append(255-milieu)
#             H_x.append(x*i+int(x/2))
#             H_y.append(j*y+int(y/2))  
# plt.imshow(I)
# plt.title('grille')
# plt.show()

            
        
            
            



