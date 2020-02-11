import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt


import pdb

def nothing(x):
    pass

def redim(factor, img):
    width = int(img.shape[1]*factor/100)
    height = int(img.shape[0]*factor/100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def histogram(output):

    Int_hist = cv2.calcHist([output], [0], None, [256], [0,256])
    b_hist = cv2.calcHist([output], [1], None, [256], [0,256])
    g_hist = cv2.calcHist([output], [2], None, [256], [0,256])
    r_hist = cv2.calcHist([output], [3], None, [256], [0,256])
    
    return [Int_hist, b_hist, g_hist, r_hist]
    
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img

#Low red and high blue
cv2.createTrackbar('Low HUE','image',0, 255, nothing)
cv2.createTrackbar('High HUE','image', 0, 255, nothing)
#Low red and high green
cv2.createTrackbar('Low SATURATION','image',0, 255, nothing)
cv2.createTrackbar('High SATURATION','image',0,255, nothing)
#Low red and high red
cv2.createTrackbar('Low VALUE','image', 0,255, nothing)
cv2.createTrackbar('High VALUE','image', 0,255, nothing)
    
#Llamamos a la imagen    
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg
img = cv2.imread(args['image'])
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = redim(5.0, img)
img2 = redim(5.0, img2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img
    
#pdb.set_trace()

while(1):
    cv2.imshow('image1', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

    
    #CONTROLADORES
    lowH = cv2.getTrackbarPos('Low HUE', 'image')
    highH = cv2.getTrackbarPos('High HUE', 'image')
    lowS = cv2.getTrackbarPos('Low SATURATION', 'image')
    highS = cv2.getTrackbarPos('High SATURATION', 'image')
    lowV = cv2.getTrackbarPos('Low VALUE', 'image')
    highV = cv2.getTrackbarPos('High VALUE', 'image')
    
    lower_bgr = np.array([lowH, lowS, lowV], dtype = "uint8")
    higher_bgr = np.array([highH, highS, highV], dtype = "uint8")
    
    img3 = img.copy()#copiamos la imagen para hacer todo
    mask = cv2.inRange(img3, lower_bgr, higher_bgr)#creamos una mascara con el inRange, que me da solo los pixeles en el rango que se requiere
    output = cv2.bitwise_and(img2, img2, mask = mask)#Creamos la imagen usando la mascara
    
    #PARA LA GENERACION DE CONTORNOS
    
    #output2 = output.copy()#Copiamos la imagen resultante y generamos un contorno al rededor
    #ret1,thresh = cv2.threshold(mask,127,255,0)#Separacion de la figura a encerrar en el contorno
    #_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#Encontramos los contornos en el treshold
    #cv2.drawContours(output2, contours, -1, (0,255,0), 1)#Dibujamos el contorno en la copia de la imagen resultante
    #pdb.set_trace()
    
    #MOSTRAR IMAGENES
    
    #pdb.set_trace()
    cv2.imshow('image2', output)
    #cv2.imshow('image3', output2) #Mostramos la imagen
    #v2.imshow('output', output)
    #plt.imshow(img)

    
cv2.destroyAllWindows()#Destruimos todas las ventanas