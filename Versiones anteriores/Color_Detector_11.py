#ESTE CODIGO NO ESTA OPTIMIZADO, Hay procesos duplicados


import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy import exp, pi
import pdb

def nothing(x):
    pass

def nonzero(img, lum, H, S,V):
    
    countimg = cv2.countNonZero(img)
    countlum = cv2.countNonZero(lum)
    countH = cv2.countNonZero(H)
    countS = cv2.countNonZero(S)
    countV = cv2.countNonZero(V)
    
    return [countimg, countlum, countH,countS,countV]

def MeanAndSigma(x, y):
    SumY = 0
    for j in y:
        SumY += j
    
    if SumY != 0:#Evitamos divisiones entre cero
        SumM = []
        for (i,j) in zip(x,y):
            SumM.append(i*j)
            
        mean = sum(SumM)/SumY #Calculamos el promedio pesado con los y
        
        SumSD = []
        for (i,j) in zip(x,y):
            SumSD.append(j * (i - mean)**2)#calculamos la desviacion estandar
        sigma = np.sqrt(sum(SumSD))/SumY

    else:
        mean = 0
        sigma = 0
    
    return [mean, sigma]

def redim(factor, img):#Funcion que redimenciona la imagen
    width = int(img.shape[1]*factor/100)
    height = int(img.shape[0]*factor/100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def DivIMG(img):#Dvidimos la imagen en HSV
    Lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Hacemos una copia de la imagen pero en blanco y negro
    H = img[:,:,0]#Separacion de canal H
    S = img[:,:,1]#Separacion de canal S
    V = img[:,:,2]#Separacion de canal V

    return [Lum, H, S, V]

def hist(img, Lum, H, S, V, mask):
    #Calculamos los histogramas de cada canal y el de la intensidad total para graficacion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Cambiamos la uagen a blanco y negro
    imgh = cv2.calcHist([img], [0], None, [256], [0,256]) #Calculamos el histograma de la imagen total, sin aplicar mascaras
    Ih = cv2.calcHist([Lum], [0], mask, [256], [0,256])#Calculamos el histograma de lum aplicando la mascara (Lum es la imagen pero aplicando la mascara)
    Hh = cv2.calcHist([H], [0], mask, [256], [0,256])#Lo mismo del anterio pero solo al canal H
    Sh = cv2.calcHist([S], [0], mask, [256], [0,256])#Lo mismo del anterio pero solo al canal S
    Vh = cv2.calcHist([V], [0], mask, [256], [0,256])#Lo mismo del anterio pero solo al canal v
  
  
  #Esta parte no hace falta, pasa el formato de la lista de [[],[],[],[],[],[]...] a [,,,,,,]  
    for i in range(len(Ih)):
        imgh[i] = imgh[i][0]
        Ih[i] = Ih[i][0]
        Hh[i] = Hh[i][0]
        Sh[i] = Sh[i][0]
        Vh[i] = Vh[i][0]
    #Elimiamos la cuenta de valor de intencidad cero, ya que es un pico que no queremos
    imgh[0] = 0
    Ih[0] = 0
    Hh[0] = 0
    Sh[0] = 0
    Vh[0] = 0
    
    
    #Calculamos los maximos de las listas anterores, solo esta siendo usado Maximgh
    maximgh = max(imgh)
    maxIh = max(Ih)
    maxHh = max(Hh)
    maxSh = max(Sh)
    maxVh = max(Vh)
    
    #Normalizamos las listas con respecto al valor maximgh
    imghN = []
    IhN = []
    HhN = []
    ShN = []
    VhN = []
    
    for i in range(len(imgh)):
        imghN.append(imgh[i]/maximgh)   
    for i in range(len(Ih)):
        #IhN.append(Ih[i]/maxIh)
        IhN.append(Ih[i]/maximgh)
    for i in range(len(Hh)):
        #bhN.append(bh[i]/maxbh)
        HhN.append(Hh[i]/maximgh) 
    for i in range(len(Sh)):
        #ghN.append(gh[i]/maxgh)
        ShN.append(Sh[i]/maximgh)
    for i in range(len(Vh)):
        #rhN.append(rh[i]/maxrh)
        VhN.append(Vh[i]/maximgh)        
    
    #pdb.set_trace()
    return [imghN, IhN, HhN, ShN, VhN]

def sums(img, output, mask): #Analisis de porcentajes
    
    img_H = img[:,:,0]
    img_S = img[:,:,1]
    img_V = img[:,:,2]
    img_T = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    output_H = output[:,:,0]
    output_S = output[:,:,1]
    output_V = output[:,:,2]
    output_T = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    #hacemos el conteo de los pixeles no nulos de la imagen total
    img_h = cv2.calcHist([img_T], [0], None, [256], [0,256]) 
    img_Hh = cv2.calcHist([img_H], [0], None, [256], [0,256])
    img_Sh = cv2.calcHist([img_S], [0], None, [256], [0,256])
    img_Vh = cv2.calcHist([img_V], [0], None, [256], [0,256])
    
    #hacemos el conteo de los pixeles no nulos de la imagen analizada
    out_h = cv2.calcHist([output_T], [0], mask, [256], [0,256]) 
    out_Hh = cv2.calcHist([output_H], [0], mask, [256], [0,256])
    out_Sh = cv2.calcHist([output_S], [0], mask, [256], [0,256])
    out_Vh = cv2.calcHist([output_V], [0], mask, [256], [0,256])
    #pdb.set_trace()
    
    img_sumT = 0
    img_sumH = 0
    img_sumS = 0
    img_sumV = 0
    out_sumT = 0
    out_sumH = 0
    out_sumS = 0
    out_sumV = 0

##Esta parte no hace falta, pasa el formato de la lista de [[],[],[],[],[],[]...] a [,,,,,,]  
    for i in range(len(img_h)):
        img_h[i] = img_h[i][0]
        out_h[i] = out_h[i][0]
        img_Hh[i] = img_Hh[i][0]
        out_Hh[i] = out_Hh[i][0]
        img_Sh[i] = img_Sh[i][0]
        out_Sh[i] = out_Sh[i][0]
        img_Vh[i] = img_Vh[i][0]
        out_Vh[i] = out_Vh[i][0]
    
    #eliminamo la cuenta de pixeles de valor cero
    img_h[0] = 0
    out_h[0] = 0
    img_Hh[0] = 0
    out_Hh[0] = 0
    img_Sh[0] = 0
    out_Sh[0] = 0
    img_Vh[0] = 0
    out_Vh[0] = 0
    
    #Hacemos la suma total de pixeles 
    for i in range(len(img_h)):
        img_sumT += img_h[i]
        out_sumT += out_h[i]
        img_sumH += img_Hh[i]
        out_sumH += out_Hh[i]
        img_sumS += img_Sh[i]
        out_sumS += out_Sh[i]
        img_sumV += img_Vh[i]
        out_sumV += out_Vh[i]
    #pdb.set_trace()   
    return [img_sumT, img_sumH, img_sumS, img_sumV, out_sumT, out_sumH, out_sumS, out_sumV]

#FUNCION DE COMPARACION          
def comparition(img_sumT, img_sumH, img_sumS, img_sumV, out_sumT, out_sumH, out_sumS, out_sumV):
    
    #LAS CONDICIONES SON PARA VEITAR LA DIVISION POR CERO
    if img_sumT != 0:    
        Tpixels_percent = (out_sumT/img_sumT)*100
    else:
        Tpixels_percent = 'No comparition'
    if img_sumH != 0:
        Hpixels_percent = (out_sumH/img_sumH)*100
    else:
        Hpixels_percent = 'No comparition'
    if img_sumS != 0:    
        Spixels_percent = (out_sumS/img_sumS)*100
    else:
        Spixels_percent = 'No comparition'
    if img_sumV != 0:
        Vpixels_percent = (out_sumV/img_sumV)*100
    else: 
        Vpixels_percent = 'No comparition'
    
    return [Tpixels_percent, Hpixels_percent, Spixels_percent, Vpixels_percent] 




#VENTANA DE CONTROLADORES    
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#CREACION DE CONTROLADORES
#Low red and high blue
cv2.createTrackbar('Low Hue','image',0, 180, nothing)
cv2.createTrackbar('High Hue','image', 0, 180, nothing)
#Low red and high green
cv2.createTrackbar('Low Saturation','image',0, 255, nothing)
cv2.createTrackbar('High Saturation','image',0,255, nothing)
#Low red and high red
cv2.createTrackbar('Low Value','image', 0,255, nothing)
cv2.createTrackbar('High Value','image', 0,255, nothing)
#---------------------------------------------------------------
    
#LECTURA DE LA IMAGEN  
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg

#---------------------------------------------------------------

#Redimencionamos la imagen
img = cv2.imread(args['image'])
img2 = img.copy()
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#A partir de esta imagen creamos la mascara con los rangos

Rfactor = 3.0

img = redim(Rfactor, img)
img2 = redim(Rfactor, img2)
imgHSV = redim(Rfactor, imgHSV)
shape = img.shape[0]*img.shape[1]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Creamos la ventana para mistras a img


#-------------------------------------------------------------------

#CREACION DEL ESPACIO DE GRAFICACION 1
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.ion()

line1, = ax1.plot(np.linspace(0, 1, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
line2, = ax1.plot(np.linspace(0, 1, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
line3, = ax1.plot(np.linspace(0, 1, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
line4, = ax1.plot(np.linspace(0, 1, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
line5, = ax1.plot(np.linspace(0, 1, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio

ax1.set_xlim(0,256)#limite de numeracion del eje x
ax1.set_ylim([0, 1.1])#Limite de eje y
ax1.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax1.set_title('Histograms Lum, H, S, V')
ax1.set_xlabel('Intensity', color = 'black')
ax1.set_ylabel('Counts', color = 'black')

#CREACION DEL ESPACIO DE GRAFICACION 2

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.ion()
plt.autoscale(enable=True, tight = True)
line6, = ax2.plot(np.linspace(0, 1, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
line7, = ax2.plot(np.linspace(0, 1, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
#pdb.set_trace()

ax2.set_xlim(0,256)#limite de numeracion del eje x
ax2.set_ylim([0, 1.1])#Limite de eje y
ax2.set_title('Histograms fits')
ax2.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax2.set_xlabel('Intensity', color = 'black')
ax2.set_ylabel('Counts', color = 'black')

plt.show(block=False)
fig1.canvas.draw()
fig2.canvas.draw()

#---------------------------------------------------------------------------------

#LOOP DE EJECUCION
while(1):
    cv2.imshow('img', img)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc
    
    #CONTROLADORES para control en vivo
    lowH = cv2.getTrackbarPos('Low Hue', 'image')
    highH = cv2.getTrackbarPos('High Hue', 'image')
    lowS = cv2.getTrackbarPos('Low Saturation', 'image')
    highS = cv2.getTrackbarPos('High Saturation', 'image')
    lowV = cv2.getTrackbarPos('Low Value', 'image')
    highV = cv2.getTrackbarPos('High Value', 'image')
    #Arrays de los valores de los controladores
    lower_HSV = np.array([lowH, lowS, lowV], dtype = "uint8")
    higher_HSV = np.array([highH, highS, highV], dtype = "uint8")
    
    mask = cv2.inRange(imgHSV, lower_HSV, higher_HSV)#creamos una mascara con el inRange, que me da solo los pixeles en el rango que se requiere
    output = cv2.bitwise_and(img2, img2, mask = mask)#Se la palicamos a img la imagen original
    #Creamos la imagen usando la mascara

    #ANALISIS DE IMAGEN OUTPUT
    if np.count_nonzero(output) != 0:#EStablecemos la condicion para asegurar que los histogramas solo se calculen cuando hay pixeles en rango
        
        [img_sumT, img_sumH, img_sumS, img_sumV, out_sumT, out_sumH, out_sumS, out_sumV] = sums(img, output, mask)
        
        [Tpixels_percent, Hpixels_percent, Spixels_percent, Vpixels_percent] = comparition(img_sumT, 
                                                                                           img_sumH, 
                                                                                           img_sumS, 
                                                                                           img_sumV, 
                                                                                           out_sumT, 
                                                                                           out_sumH, 
                                                                                           out_sumS, 
                                                                                           out_sumV)
        
        #print('Percent pixels in BGR range: ' + str(Tpixels_percent))
        
        #DIVISION DE IMAGEN EN CANALES Y LUMINOSIDAD Y CALCULO DE HISTOGRAMAS
        [Lum, H, S, V] = DivIMG(output)
        [img_hist, Int_hist, H_hist, S_hist, V_hist] = hist(img, Lum, H, S, V, mask)
        #print(Int_hist)
        line1.set_ydata(Int_hist)#utiliza los datos de hist_mascara para realizar la grafica
        line2.set_ydata(H_hist)#utiliza los datos de hist_mascara para realizar la grafica
        line3.set_ydata(S_hist)
        line4.set_ydata(V_hist)
        line5.set_ydata(img_hist)
        
        line1.set_label('Luminosity')
        line2.set_label('Hue channel')
        line3.set_label('Saturation channel')
        line4.set_label('Value channel')
        line5.set_label('Black White image')
        
        line1.set_color('gray')
        line2.set_color('blue')
        line3.set_color('green')
        line4.set_color('red')
        line5.set_color('black')
        
        
        ax1.legend(loc='best')
        ax1.draw_artist(ax1.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax1.draw_artist(line1) #Redibuja line solo si es necesario
        fig1.canvas.draw()#updatea el grafico con la nueva curva
        fig1.canvas.flush_events()#Hace un sleep para que se pueda crear la grafica
        
        
        line6.set_ydata(Int_hist)#utiliza los datos de hist_mascara para realizar la grafica
        line7.set_ydata(img_hist)
        line6.set_color('black')
        line6.set_label('Luminosity of seccion')
        line7.set_color('gray')
        line7.set_label('Luminosity image')
         
        ax2.legend(loc='best')
        ax2.draw_artist(ax2.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax2.draw_artist(line6) #Redibuja line solo si es necesario
        fig2.canvas.draw()#updatea el grafico con la nueva curva
        fig2.canvas.flush_events()#Hace un sleep para que se pueda crear la grafica
    
    #PARA LA GENERACION DE CONTORNOS
    output2 = output.copy()#Copiamos la imagen resultante y generamos un contorno al rededor
    ret1,thresh = cv2.threshold(mask,127,255,0)#Separacion de la figura a encerrar en el contorno
    #_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#Encontramos los contornos en el treshold
    #DESCOMENTAR LA LINEA SIGUIENTE Y COMENTAR LA LINEA ANTERIOR SI SE EJECUTA EN MAC
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#Encontramos los contornos en el treshold
    cv2.drawContours(output2, contours, -1, (0,255,0), 1)#Dibujamos el contorno en la copia de la imagen resultante
    
    #MOSTRAR IMAGENES
    #pdb.set_trace()
    
    cv2.imshow('image2', mask)
    cv2.imshow('image3', output) #Mostramos la imagen


    if k == ord(' '):
        print('Percent pixels in HSV range: ' + str(Tpixels_percent))
        print('Percent pixels in H channel: ' + str(Hpixels_percent))
        print('Percent pixels in S channel: ' + str(Spixels_percent))
        print('Percent pixels in V channel: ' + str(Vpixels_percent))

            
    
cv2.destroyAllWindows()#Destruimos todas las ventanas