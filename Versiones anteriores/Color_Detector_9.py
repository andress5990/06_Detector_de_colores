import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy import exp, pi
import pdb

def nothing(x):
    pass

def nonzero(img, lum, b, g, r):
    
    countimg = cv2.countNonZero(img)
    countlum = cv2.countNonZero(lum)
    countb = cv2.countNonZero(b)
    countg = cv2.countNonZero(g)
    countr = cv2.countNonZero(r)
    
    return [countimg, countlum, countb,countg,countr]

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

def redim(factor, img):
    width = int(img.shape[1]*factor/100)
    height = int(img.shape[0]*factor/100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def DivIMG(img):
    
    Lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    return [Lum, b, g, r]

def hist(img, Lum, b, g, r, mask):
    #Calculamos los histogramas de cada canal y el de la intensidad total para graficacion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgh = cv2.calcHist([img], [0], None, [256], [0,256]) 
    Ih = cv2.calcHist([Lum], [0], mask, [256], [0,256])
    bh = cv2.calcHist([b], [0], mask, [256], [0,256])
    gh = cv2.calcHist([g], [0], mask, [256], [0,256])
    rh = cv2.calcHist([r], [0], mask, [256], [0,256])
    
    for i in range(len(Ih)):
        imgh[i] = imgh[i][0]
        Ih[i] = Ih[i][0]
        bh[i] = bh[i][0]
        gh[i] = gh[i][0]
        rh[i] = rh[i][0]
    
    maximgh = max(imgh)
    maxIh = max(Ih)
    maxbh = max(bh)
    maxgh = max(gh)
    maxrh = max(rh)
    
    imghN = []
    IhN = []
    bhN = []
    ghN = []
    rhN = []
    
    for i in range(len(imgh)):
        imghN.append(imgh[i]/maximgh)   
    for i in range(len(Ih)):
        #IhN.append(Ih[i]/maxIh)
        IhN.append(Ih[i]/maximgh)
    for i in range(len(bh)):
        #bhN.append(bh[i]/maxbh)
        bhN.append(bh[i]/maximgh) 
    for i in range(len(gh)):
        #ghN.append(gh[i]/maxgh)
        ghN.append(gh[i]/maximgh)
    for i in range(len(rh)):
        #rhN.append(rh[i]/maxrh)
        rhN.append(rh[i]/maximgh)        
    
    #pdb.set_trace()
    return [imghN, IhN, bhN, ghN, rhN]

def sums(img, output, mask): #Analisis de porcentajes
    
    img_b = img[:,:,0]
    img_g = img[:,:,1]
    img_r = img[:,:,2]
    
    output_b = output[:,:,0]
    output_g = output[:,:,1]
    output_r = output[:,:,2]
    
    #hacemos el conteo de los pixeles no nulos de la imagen total
    img_h = cv2.calcHist([img_b], [0], None, [256], [0,256]) 
    img_bh = cv2.calcHist([img_b], [0], None, [256], [0,256])
    img_gh = cv2.calcHist([img_g], [0], None, [256], [0,256])
    img_rh = cv2.calcHist([img_r], [0], None, [256], [0,256])
    
    #hacemos el conteo de los pixeles no nulos de la imagen analizada
    out_h = cv2.calcHist([output], [0], mask, [256], [0,256]) 
    out_bh = cv2.calcHist([output_b], [0], mask, [256], [0,256])
    out_gh = cv2.calcHist([output_g], [0], mask, [256], [0,256])
    out_rh = cv2.calcHist([output_r], [0], mask, [256], [0,256])
    
    img_sumT = 0
    img_sumb = 0
    img_sumg = 0
    img_sumr = 0
    out_sumT = 0
    out_sumb = 0
    out_sumg = 0
    out_sumr = 0

    for i in range(len(img_h)):
        img_h[i] = img_h[i][0]
        out_h[i] = out_h[i][0]
        img_bh[i] = img_bh[i][0]
        out_bh[i] = out_bh[i][0]
        img_gh[i] = img_gh[i][0]
        out_gh[i] = out_gh[i][0]
        img_rh[i] = img_rh[i][0]
        out_rh[i] = out_rh[i][0]
    
    #eliminamo la cuenta de pixeles de valor cero
    img_h[0] = 0
    out_h[0] = 0
    img_bh[0] = 0
    out_bh[0] = 0
    img_gh[0] = 0
    out_gh[0] = 0
    img_rh[0] = 0
    out_rh[0] = 0
    
    
    for i in range(len(img_h)):
        img_sumT += img_h[i]
        out_sumT += out_h[i]
        img_sumb += img_bh[i]
        out_sumb += out_bh[i]
        img_sumg += img_gh[i]
        out_sumg += out_gh[i]
        img_sumr += img_rh[i]
        out_sumr += out_rh[i]
        
    return [img_sumT, img_sumb, img_sumg, img_sumr, out_sumT, out_sumb, out_sumg, out_sumr]

#FUNCION DE COMPARACION          
def comparition(img_sumT, img_sumb, img_sumg, img_sumr, out_sumT, out_sumb, out_sumg, out_sumr):
    
    #LAS CONDICIONES SON PARA VEITAR LA DIVISION POR CERO
    if img_sumT != 0:    
        Tpixels_percent = (out_sumT/img_sumT)*100
    else:
        Tpixels_percent = 'No comparition'
    if img_sumb != 0:
        Bpixels_percent = (out_sumb/img_sumb)*100
    else:
        Bpixels_percent = 'No comparition'
    if img_sumg != 0:    
        Gpixels_percent = (out_sumg/img_sumg)*100
    else:
        Gpixels_percent = 'No comparition'
    if img_sumr != 0:
        Rpixels_percent = (out_sumr/img_sumr)*100
    else: 
        Rpixels_percent = 'No comparition'
    
    return [Tpixels_percent, Bpixels_percent, Gpixels_percent, Rpixels_percent] 




#VENTANA DE CONTROLADORES    
cv2.namedWindow('image', cv2.WINDOW_NORMAL)


#CREACION DE CONTROLADORES
#Low red and high blue
cv2.createTrackbar('Low Blue','image',0, 255, nothing)
cv2.createTrackbar('High Blue','image', 0, 255, nothing)
#Low red and high green
cv2.createTrackbar('Low Green','image',0, 255, nothing)
cv2.createTrackbar('High Green','image',0,255, nothing)
#Low red and high red
cv2.createTrackbar('Low Red','image', 0,255, nothing)
cv2.createTrackbar('High Red','image', 0,255, nothing)
    
#LECTURA DE LA IMAGEN  
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg

img = cv2.imread(args['image'])
img = redim(5.0, img)
shape = img.shape[0]*img.shape[1]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img

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
ax1.set_title('Histograms Lum, b, g, r')
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

#LOOP DE EJECUCION
while(1):
    cv2.imshow('image1', img)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc
    
    #CONTROLADORES
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
    #Creamos la imagen usando la mascara

    #ANALISIS DE IMAGEN OUTPUT
    if np.count_nonzero(output) != 0:#EStablecemos la condicion para asegurar que los histogramas solo se calculen cuando hay pixeles en rango
        
        [img_sumT, img_sumb, img_sumg, img_sumr, out_sumT, out_sumb, out_sumg, out_sumr] = sums(img, output, mask)
        
        [Tpixels_percent, Bpixels_percent, Gpixels_percent, Rpixels_percent] = comparition(img_sumT, 
                                                                                           img_sumb, 
                                                                                           img_sumg, 
                                                                                           img_sumr, 
                                                                                           out_sumT, 
                                                                                           out_sumb, 
                                                                                           out_sumg, 
                                                                                           out_sumr)
        
        #print('Percent pixels in BGR range: ' + str(Tpixels_percent))
        
        #DIVISION DE IMAGEN EN CANALES Y LUMINOSIDAD Y CALCULO DE HISTOGRAMAS
        [Lum, b, g, r] = DivIMG(output)
        [img_hist, Int_hist, b_hist, g_hist, r_hist] = hist(img, Lum, b, g, r, mask)
        #print(Int_hist)
        line1.set_ydata(Int_hist)#utiliza los datos de hist_mascara para realizar la grafica
        line2.set_ydata(b_hist)#utiliza los datos de hist_mascara para realizar la grafica
        line3.set_ydata(g_hist)
        line4.set_ydata(r_hist)
        line5.set_ydata(img_hist)
        
        line1.set_label('Luminosity')
        line2.set_label('Blue channel')
        line3.set_label('Green channel')
        line4.set_label('Red channel')
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
    cv2.imshow('image2', output)
    cv2.imshow('image3', output2) #Mostramos la imagen

    if k == ord(' '):
        print('Percent pixels in BGR range: ' + str(Tpixels_percent))
        print('Percent pixels in B channel: ' + str(Bpixels_percent))
        print('Percent pixels in G channel: ' + str(Gpixels_percent))
        print('Percent pixels in R channel: ' + str(Rpixels_percent))
            
    
cv2.destroyAllWindows()#Destruimos todas las ventanas