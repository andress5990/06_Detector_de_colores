import numpy as np
import argparse
import cv2 


def nothing(x):
    pass

#leemos el archvo con la terminal
img_file = argparse.ArgumentParser()
img_file.add_argument('-i', '--image', help = 'path to the image')
args = vars(img_file.parse_args())
#se llama asi
# python Color_Detector_4.py --image Space-Wallpaper-Tumblr-Awesome-E25.jpg 
img = cv2.imread(args['image'])

scale_percent =  10.0
width = int(img.shape[1]* scale_percent/100)
height = int(img.shape[0]* scale_percent/100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#El orden de la imagen cuando se hable es BGR
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Convertimos la imagen a RGB
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Creamos la ventana

ranges = [([17,15,100],[50,56,200]),
          ([49,50,50],[80,255,255]),
          ([86,31,4],[220,88,50])]

#ranges = BGRA

colors = ('b', 'g', 'r')

for (lower, upper) in ranges:#hacemos el proceso para cada color
    low = np.array(lower, dtype = 'uint8')#Convertimos cada limite en un np.array
    up = np.array(upper, dtype = "uint8")#Convertimos cada limite en un np.array
    
    #Creamos una  mascara
    mask = cv2.inRange(img, low, up)#Esta mascara separa los colores de la image
                                    #usando los valores de los rangos
    output = cv2.bitwise_and(img, img, mask = mask)#aplicamos la mascara para crear una nueva imagen 
                                                   #con los pixeles separados
                                                   

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    output = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
                                                   
    cv2.imshow('image', img)
    cv2.imshow('selection', output)
    cv2.imwrite('IMAGE.jpg', output) 

    cv2.waitKey(0)
    
    
    
