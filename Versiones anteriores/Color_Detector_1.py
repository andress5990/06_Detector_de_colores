#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt


import pdb

def nothing(x):
    pass


#fil_img = str(sys.argv[1])

#img = cv2.imread(fil_img)#Lee la imagen a color en BGR
#cv2.IMREAD_UNCHANGED es equevalente a poner un -1 

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg
img = cv2.imread(args['image'])

#Estan en el orden RGB: La imagen se ve de colores invertido porque ya esta en rgb
#En BGR se ve la imagen normal
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#convertimos a RGB
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
 
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img

while(1):
    cv2.imshow('image',img) #Mostramos la imagen
    #plt.imshow(img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas