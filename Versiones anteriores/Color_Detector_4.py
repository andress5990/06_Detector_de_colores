import numpy as np
import argparse
import cv2


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
#R = img[:,:,0]
#G = img[:,:,1]
#B = img[:,:,2]
 
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img

	
# define the list of boundaries
#boundaries = [
#	([17, 15, 100], [50, 56, 200]),
#	([86, 31, 4], [220, 88, 50]),
#	([25, 146, 190], [62, 174, 250]),
#	([103, 86, 65], [145, 133, 128])
#]
#Estos son , rojo, azul, amarillo, gris

boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([49, 50, 50], [80, 255, 255]),
	([86, 31, 4], [220, 88, 50])
]#RGB



#Son valores rango para rojo(low, up), verde(low, up) y azul(low, up)	


# Recorremos la imagen tantas veces como tuplas tengamos en los limites
for (lower, upper) in boundaries:
	#Creamos arrays de numpy de lo
	low = np.array(lower, dtype = "uint8")
	up = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(img, low, up)
	output = cv2.bitwise_and(img, img, mask = mask)
 
	#Mostramos la imagenes en un solo recuadro
	cv2.imshow("images", np.hstack([img, output]))
	cv2.waitKey(0)


#while(1):
#    cv2.imshow('image',img) #Mostramos la imagen
#    #plt.imshow(img)
#
#    k = cv2.waitKey(1) & 0xFF
#    if k == 27:#esc
#        #pdb.set_trace()
#        break# hace,el break para para el programa si se estripa esc

#cv2.destroyAllWindows()#Destruimos todas las ventanas