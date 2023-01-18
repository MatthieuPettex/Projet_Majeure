import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import PIL





def assignlabel(x1,x2,image,label_map):
    #x1 et x2 contiennent les coordonnées des points appellés
    if label_map[x2] == 0 and dist(x1,x2,image)<1:
        label_map[x2]=label_map[x1]
        labelize_neighboors(x2,image,label_map)
    return label_map


def dist(x1,x2,image):
    #x1 et x2 contiennent les coordonnées des points appellés

    #image est l'image en question

    #dist renvoie la distance entre les deux points

    return np.sqrt((image[x1][0]-image[x2][0])**2+(image[x1][1]-image[x2][1])**2+(image[x1][2]-image[x2][2])**2)

def labelize_neighboors(point,image,label_map):
    width=image.shape[1]-1
    height=image.shape[0]-1
    x=point[0]
    y=point[1]
    if label_map[min(x+1,height),y]==0 and min(x+1,height) == x+1:

        label_map = assignlabel((x,y),(x+1,y),image,label_map)

    if label_map[max(x-1,0),y]==0 and max(x-1,0) == x-1:

        label_map = assignlabel((x,y),(x-1,y),image,label_map)

    if label_map[x,min(y+1,height)]==0 and (min(y+1,height)==y+1):

        label_map = assignlabel((x,y),(x,y+1),image,label_map)

    if label_map[x,max(y-1,0)]==0 and max(y-1,0)==y-1:

        label_map=assignlabel((x,y),(x,y-1),image,label_map)

    if label_map[min(x+1,height),min(y+1,height)]==0 and (min(x+1,height) == x+1 and min(y+1,height)==y+1):

        label_map=assignlabel((x,y),(x+1,y+1),image,label_map)

    if label_map[max(x-1,0),max(y-1,0)]==0 and (max(x-1,0) == x-1 and max(y-1,0)==y-1):

        label_map=assignlabel((x,y),(x-1,y-1),image,label_map)

    if label_map[min(x+1,height),max(y-1,0)]==0 and (min(x+1,height) == x+1 and max(y-1,0)==y-1):

        label_map=assignlabel((x,y),(x+1,y-1),image,label_map)

    if label_map[max(x-1,0),min(y+1,height)]==0 and (max(x-1,0) == x-1 and min(y+1,height)==y+1):

        label_map=assignlabel((x,y),(x-1,y+1),image,label_map)
        


def labelize(image,label_map):
    width=image.shape[1]
    height=image.shape[0]
    label_value=1
    label_map[7,0]=label_value
    labelize_neighboors((7,0),image,label_map)
    for x in range(height):
        for y in range(width):
            if label_map[x,y]==0:
                label_value+=1
                label_map[x,y]=label_value
                labelize_neighboors((x,y),image,label_map)


    return label_map

def clamp(n, smallest, largest):

    return max(smallest, min(n, largest))

def direction_of_line(labels,i,j,u,v):
    width, height= np.shape(labels)
    width=-1
    height=-1
    for a in range(-1,2):
        for b in range(-1,2):
            if labels[i,j]==labels[clamp((i+a),0,width),clamp((j+b),0,height)] and a!=u and b!=v:
                dir1 , dir2 = a,b
    return dir1, dir2

def Neighboors(labels,i,j):
    width, height= np.shape(labels)
    width=-1
    height=-1
    voisins=0
    for a in range(-1,2):
        for b in range(-1,2):
            if labels[i,j]==labels[clamp((i+a),0,width),clamp((j+b),0,height)]:
                voisins+=1
    return voisins

def checklength(labels,i,j,u,v,strength):
    if Neighboors(labels,i,j)==2 :
        dir1, dir2 = direction_of_line(labels,i,j,u,v)
        strength = checklength(labels,i+dir1,j+dir2,-dir1,-dir2,strength)
        strength+=1
    return strength


#Fonction de vérification de la longueur des ensembles associés dans le cas de deux diagonales intercroisées pour un point lors de la création de la matrice de liens
def checklinkstrength(labels,i,j,u,v):
    width, height= np.shape(labels)
    width_clamp=width-1
    height_clamp=height-1
    strength_line=1
    strength_square=0
    strength_island=0
    ensemble=0
    #On commence par calculer le vote attribué à la présence d'un ensemble dans un espace autour de la connection
    for x in range(-4,4):
        for y in range(-4,4):
            if clamp((i+x),0,width_clamp)== i+x and clamp((j+y),0,height_clamp)== j+y: #si on est dans l'image
                if labels[i,j]==labels[(i+x),(j+y)]:
                    ensemble+=1

    #La valeur exacte des vote attribué est détérminé de façon empirique
    strength_square=5-ensemble/16


    #On regarde ensuite le vote attribué à la présence d'un ensemble isolé

    voisinspt1=Neighboors(labels,i,j)
    if voisinspt1==1:
        strength_island+=1
    
    if clamp((i+u),0,width_clamp)!= i+u or clamp((j+v),0,height_clamp)!= j+v:
        voisinspt2=0
    else:
        voisinspt2=Neighboors(labels,(i+u),(j+v))
        if voisinspt2==1:
            strength_island+=1

    strength_island=strength_island*5

    #On regarde enfin le vote attribué à la présence d'un ensemble sur une ligne

    strength_line+=checklength(labels,i,j,u,v,0)

    if clamp((i+u),0,width_clamp)!=(i+u):
        print(i+u)
    strength_line+=checklength(labels,clamp((i+u),0,width_clamp),clamp((j+v),0,height_clamp),-u,-v,0)



    strength_island=strength_island*5

    
    return strength_square+strength_line+strength_island
    
def create_links(labels):
    width, height= np.shape(labels)
    links=np.full((width,height,3,3), 0)
    width_clamp=width-1
    height_clamp=height-1
    for i in range(width):
        for j in range(height):
            
            links[i,j][1,1]=labels[i,j]


    for i in range(width):
       for j in range(height):
            for u in range(-1,2):
                for v in range(-1,2):
                    #si la connection est dans l'image
                    if clamp((i+u),0,width_clamp)== i+u and clamp((j+v),0,height_clamp)== j+v:
                    #si les labels des pixel et du voisin u,v sont les mêmes
                        if links[i,j][1,1]==links[i+u,j+v][1,1] and not (u==0 and v==0):

                            #Condition de vérification en cas d'entrecroisements de zones 
                            if np.abs(u)==np.abs(v) and (links[i+u,j][1,1]==links[i,j+v][1,1] and links[i+u,j][1,1]!=links[i,j][1,1]):
                                #On vérifie la force de la connection

                                if((checklinkstrength(labels,i,j,u,v)+checklinkstrength(labels,i+u,j+v,-u,-v))>(checklinkstrength(labels,i+u,j,-u,v)+checklinkstrength(labels,i,j+v,u,-v))):
                                        links[i,j][u+1,v+1]=labels[i,j]
                            else:   #On affecte la valeur 1 (il y a une ligne)
                                links[i,j][u+1,v+1]=labels[i,j]
                            
    return links

def non_linked_Labelisation(nom, pixel_size):
    image1 = Image.open(nom)
    image = np.array(image1)

    #On divise la taille de l'image par 4*4 en gardant la valeur d'un point correspondant car l'image est en 64*64 pour 16*16 pixels
    width_original, height_original, _ = image.shape

    selection_matrix= np.zeros((width_original, int(height_original/pixel_size)))
    for i in range(width_original):
        for j in range(int(height_original/pixel_size)):
            if i==pixel_size*j :
                selection_matrix[i, j] = 1
    image_resized=np.zeros((int(width_original/pixel_size), int(height_original/pixel_size),4))
    for k in range(4):
        image_resized[: , :, k  ] = np.dot(np.transpose(selection_matrix),np.dot(image[:,:, k ],selection_matrix))

    image_resized=image_resized.astype(np.uint8)
    labels = np.zeros(image_resized.shape[:2])
    labels = labelize(image_resized,labels)
    width, height= np.shape(labels)
    links=create_links(labels)

    #On affiche les liens
    links_shown=np.zeros((3*width,3*height))

    for i in range(width):
        for j in range (height):
            for u in range(-1,2):
                for v in range(-1,2):
                    links_shown[(3*i+u+1)%(3*width),(3*j+v+1)%(3*height)]=links[i,j][u+1,v+1]

    return links, links_shown, labels


links, links_shown, labels = non_linked_Labelisation("boo16_16.png", 1)
#On affiche les liens
image1 = Image.open("boo16_16.png")
image = np.array(image1)
plt.figure(figsize=(15,15))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(labels)
plt.subplot(1, 3, 3)
plt.imshow(links_shown)
plt.show()
