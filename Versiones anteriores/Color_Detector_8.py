import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt


import pdb

def nothing(x):
    pass

def histogram(output):

    Int_hist = cv2.calcHist([output], [0], None, [256], [0,256])
    b_hist = cv2.calcHist([output], [1], None, [256], [0,256])
    g_hist = cv2.calcHist([output], [2], None, [256], [0,256])
    r_hist = cv2.calcHist([output], [3], None, [256], [0,256])
    
    return [Int_hist, b_hist, g_hist, r_hist]
    
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img

#Low red and high blue
cv2.createTrackbar('Low Blue','image',0, 255, nothing)
cv2.createTrackbar('High Blue','image', 0, 255, nothing)
#Low red and high green
cv2.createTrackbar('Low Green','image',0, 255, nothing)
cv2.createTrackbar('High Green','image',0,255, nothing)
#Low red and high red
cv2.createTrackbar('Low Red','image', 0,255, nothing)
cv2.createTrackbar('High Red','image', 0,255, nothing)
    
#Llamamos a la imagen    
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg
img = cv2.imread(args['image'])
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img
    
#pdb.set_trace()

while(1):
    cv2.imshow('imge', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

    
    #EStablecemos los trakers
    lowB = cv2.getTrackbarPos('Low Blue', 'image')
    highB = cv2.getTrackbarPos('High Blue', 'image')
    lowG = cv2.getTrackbarPos('Low Green', 'image')
    highG = cv2.getTrackbarPos('High Green', 'image')
    lowR = cv2.getTrackbarPos('Low Red', 'image')
    highR = cv2.getTrackbarPos('High Red', 'image')
    
    lower_bgr = np.array([lowB, lowG, lowR], dtype = "uint8")
    higher_bgr = np.array([highB, highG, highR], dtype = "uint8")
    
    img2 = img.copy()#copiamos la imagen para hacer todo
    mask = cv2.inRange(img2, lower_bgr, higher_bgr)#creamos una mascara con el inRange, que me da solo los pixeles en el rango que se requiere
    output = cv2.bitwise_and(img, img, mask = mask)
    #output = cv2.bitwise_and(img, mask)
    #pdb.set_trace()
    

    #pdb.set_trace()
    cv2.imshow('image', output) #Mostramos la imagen
    #v2.imshow('output', output)
    #plt.imshow(img)

    
cv2.destroyAllWindows()#Destruimos todas las ventanas