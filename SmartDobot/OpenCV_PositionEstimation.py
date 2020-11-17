import cv2
import numpy as np
from serial.tools import list_ports
from pydobot import Dobot
import csv
import time
from PIL import Image
import ML_Regression as regression
from termcolor import colored
from twisted.internet import task, reactor

# DEFINIZIONE PARAMETRI FUNZIONAMENTO DOBOT
areaMin = 350
areaMax = 5000
min_z = 45.0

# FUNZIONALITA' DEL DOBOT
def homing(device):
    device.home()
    
def spegni_pompa_aria(device):
    device.set_io(10, False)
    device.set_io(11, False)
    device.set_io(12, False)
    device.set_io(16, False)
    
def apri_chela(device):
    device.grip(enable=False)
    time.sleep(1)
    
def chiudi_chela(device):
    device.grip(enable=True)
    time.sleep(1)

def avvia_rullo(device):
    device.conveyor_belt(1)

def ferma_rullo(device):
    device.conveyor_belt(0)
    
def stampa_posizione(device):
    pose = device.get_pose()
    position = pose.position
    print('Dobot', position)         
   
# MOVIMENTO DEL DOBOT
def scuoti():
    #SCUOTE IL BRACCIO
    posa_x = 1
    posa_y = -175.0
    posa_z = 50.0
    device.move_to(posa_x+50, posa_y, posa_z, 0.0, True)
    device.move_to(posa_x-50, posa_y, posa_z, 0.0, True)
    
def posa_oggetto():
    # POSIZIONE DI LATO
    posa_x = 1
    posa_y = -175.0
    posa_z = 50.0
    device.move_to(posa_x-20, posa_y, posa_z, 0.0, True)
    apri_chela(device)    
    scuoti()

def posizione_intermedia():
    initial_x = 200.0
    initial_y = 0.0
    initial_z = 170.0
    device.move_to(initial_x, initial_y, initial_z, 0.0, True)
    
def prima_fase_presa(x,y,r):
    device.move_to(x, y, 150.0, 15.0, True)
    time.sleep(1)
    
def seconda_fase_presa(x,y,r):
    device.move_to(x, y, 150.0, r, True)
    time.sleep(1)
    apri_chela(device)
    
def terza_fase_presa(x,y,r):
    max_z = 150.0
    device.move_to(x, y, min_z+20, r, True)
    time.sleep(1)
    device.move_to(x, y, min_z, r, True)
    chiudi_chela(device)

def prendi(x,y,r):
    device.clear_alarms()
    
    x = round(float(x), 2)
    y = round(float(y), 2)
    r = round(float(r), 2)
    posizione_intermedia()
    
    # controllo che la coordinata x calcolata non superi i limiti del Dobot
    x=x+5
    if x <= 195:
        x = 195
    elif x >= 265:
        x = 265

    # controllo che la coordinata y calcolata non superi i limiti del Dobot
    if y<0:
        y=y+2
    if y>0:
        y=y+8
    if y>80:
        y=80
    elif y<-70:
        y=-70
    
    r=15
    
    # 3 FASI PRESA
    
    prima_fase_presa(x,y,r) # mi sposto in posizione x,y dell'oggetto
    seconda_fase_presa(x,y,r) # ruoto la chela 
    terza_fase_presa(x,y,r) # abbasso il braccio
    
    chiudi_chela(device)
    posizione_intermedia()
    posa_oggetto()
        
def prendi_tutto(device, lista_contorni, regressore_x, regressore_y, regressore_r):
    for cnt in lista_contorni:
        (x, y, w, h) = cv2.boundingRect(cnt)

        rect = cv2.minAreaRect(cnt)
        punti_inizio, punti_centrali, angolo_rotazione = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)   
        area = cv2.contourArea(box)
        perimeter = cv2.arcLength(box,True)
                
        if area > areaMin and area < areaMax:
            x_centro = int(x + (w/2))
            y_centro = int(y + (w/2))
            x = regression.x_predizione(regressore_x, x_centro, y_centro)
            y = regression.y_predizione(regressore_y, x_centro, y_centro)
            r = regression.r_predizione(regressore_r, -angolo_rotazione)
            time.sleep(1)
            prendi(x,y,r)

    posizione_intermedia()
    spegni_pompa_aria(device)    
    print(colored('   Fatto!', 'yellow'))
    
def avanti_tappeto(device):
    avvia_rullo(device)
    time.sleep(0.5)
    ferma_rullo(device)

# FUNZIONI DI IMAGE PROCESSING

def trova_contorni(frame, lista_contorni):  
    belt = frame
    gray_belt = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_belt, 75, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    lista_contorni.extend(contours)
    #cv2.imshow('black&white', threshold) 
    return lista_contorni
 
def color_detection(frame):
    #cv2.imshow('originale', frame)
    frame_rosso = frame.copy()
    frame_rosa = frame.copy()
    frame_arancione = frame.copy()
    frame_giallo = frame.copy()
    frame_verde = frame.copy()
    frame_azzurro = frame.copy()
    frame_blu = frame.copy()
    frame_viola = frame.copy()
    
    frame_colori_selezionati = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #--------------------- COLORE ROSA -------------------------------------------
    low_rosa = np.array(  [0,0,70])
    high_rosa = np.array(  [60,80,255])
    rosa_mask = cv2.inRange(frame, low_rosa, high_rosa)
    
    solo_rosa = cv2.bitwise_and(frame_rosa, frame_rosa, mask=rosa_mask)
    
    # ------------------- COLORE ROSSO --------------------------------------------
    lower_red = np.array(  [0,75,75])
    upper_red = np.array(  [10,255,255])
    rosso_mask1= cv2.inRange(frame, lower_red, upper_red)

    lower_red = np.array(  [170,75,75])
    upper_red = np.array(  [180,255,255])
    rosso_mask2 = cv2.inRange(frame,lower_red,upper_red)
    # maschera rossa completa
    rosso_mask = rosso_mask1 + rosso_mask2
    solo_rosso = cv2.bitwise_and(frame_rosso, frame_rosso, mask=rosso_mask)
    
    #--------------------- COLORE ARANCIONE -------------------------------------------
    low_arancione = np.array(  [11,125,125])
    high_arancione = np.array(  [18,255,255])
    arancione_mask = cv2.inRange(frame, low_arancione, high_arancione)
    
    solo_arancione = cv2.bitwise_and(frame_arancione, frame_arancione, mask=arancione_mask)
    
    # ------------------- COLORE GIALLO --------------------------------------------
    low_giallo = np.array(  [19, 50, 50])
    high_giallo = np.array(  [35, 255, 255])
    giallo_mask = cv2.inRange(frame, low_giallo, high_giallo)
    
    solo_giallo = cv2.bitwise_and(frame_giallo, frame_giallo, mask=giallo_mask)
    
    # ------------------- COLORE VERDE --------------------------------------------
    low_verde = np.array(  [36 , 50, 50])
    high_verde = np.array(  [80, 255, 255])
    verde_mask = cv2.inRange(frame, low_verde, high_verde)
    
    solo_verde = cv2.bitwise_and(frame_verde, frame_verde, mask=verde_mask)
    
    # ------------------- COLORE AZZURRO --------------------------------------------
    low_azzurro = np.array(  [81, 0, 0])
    high_azzurro = np.array(  [100, 255, 255])
    azzurro_mask = cv2.inRange(frame, low_azzurro, high_azzurro)
    
    solo_azzurro = cv2.bitwise_and(frame_azzurro, frame_azzurro, mask=azzurro_mask)

    # ------------------- COLORE BLU --------------------------------------------
    low_blu = np.array(  [101, 125, 125])
    high_blu = np.array(  [127, 255, 255])
    blu_mask = cv2.inRange(frame, low_blu, high_blu)
    
    solo_blu = cv2.bitwise_and(frame_blu, frame_blu, mask=blu_mask)
    
    # ------------------- COLORE VIOLA --------------------------------------------
    low_viola = np.array(  [128, 50, 50])
    high_viola = np.array(  [169, 255, 255])
    viola_mask = cv2.inRange(frame, low_viola, high_viola)
    
    solo_viola = cv2.bitwise_and(frame_viola, frame_viola, mask=viola_mask)
    
    # ------------------- STAMPO E RITORNO --------------------------------------------
    #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    #cv2.imshow('rosso', solo_rosso)
    #cv2.imshow('blu', solo_blu)
    #cv2.imshow('verde', solo_verde)
    #cv2.imshow('giallo', solo_giallo)
    #cv2.imshow('rosa', solo_rosa)
    #cv2.imshow('arancione', solo_arancione)
    
    mask_totale =  rosso_mask + arancione_mask + giallo_mask + verde_mask +  blu_mask + viola_mask
    solo_colori = cv2.bitwise_and(frame_colori_selezionati, frame_colori_selezionati, mask=mask_totale)
    cv2.imshow('colori', solo_colori)
    
    return solo_rosso,solo_arancione,solo_giallo,solo_verde,solo_azzurro,solo_blu,solo_viola

def guarda_e_prendi(frame, lista_contorni):
    
    solo_rosso,solo_arancione,solo_giallo,solo_verde,solo_azzurro,solo_blu,solo_viola = color_detection(frame)
    
    lista_contorni = trova_contorni(solo_rosso, lista_contorni)
    lista_contorni = trova_contorni(solo_arancione, lista_contorni)
    lista_contorni = trova_contorni(solo_giallo, lista_contorni)
    lista_contorni = trova_contorni(solo_verde, lista_contorni)
    #lista_contorni = trova_contorni(solo_azzurro, lista_contorni)
    lista_contorni = trova_contorni(solo_blu, lista_contorni)
    lista_contorni = trova_contorni(solo_viola, lista_contorni)
    
    cv2.imshow("frame", frame)
    
    for cnt in lista_contorni:
        # contorni approssimati
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # contorni precisi
        rect = cv2.minAreaRect(cnt)
        punti_inizio, punti_centrali, angolo_rotazione = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # calcolo area e perimetro in base ai contorni precisi
        area = cv2.contourArea(box)
        perimeter = cv2.arcLength(box,True)
   
        if area > areaMin and area < areaMax:
            # trovo la forma dell'oggetto
            approx = cv2.approxPolyDP(box, 0.1 * cv2.arcLength(cnt, True), True)
            
            # calcolo il centro dell'oggetto
            x_centro = int(x + (w/2))
            y_centro = int(y + (h/2))
            
            # disegno sul frame le informazioni ricavate dai contorni
            cv2.drawContours(frame, [box],0,(0,0,255),2) # contorni                
            #cv2.putText(frame, 'area: '+str(area), (x, y-75), 1, 1, (0, 255, 0)) # scrivo l'area
            cv2.circle(frame, (int(x_centro), int(y_centro)), 2,(255, 255, 255), 2) # sx
            cv2.putText(frame, str('x: '+str(x_centro)+'| y: '+ str(y_centro)), (int(x_centro-15), int(y_centro-15)), 1, 1, (255, 255, 255)) # scrivo la forma    
            

            print(colored('   Prendo gli oggetti', 'yellow'))
            prendi_tutto(device, lista_contorni, regressore_x, regressore_y, regressore_r)
            lista_contorni =  lista_contorni.clear()
    
#-------------------------------------------------------------------------

# >>>>>>>>>>>>>>>>>>>>>>> MAIN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

sub_background = cv2.createBackgroundSubtractorKNN()

print('Cerco il Dobot...')
try:
    port = list_ports.comports()  [0].device
    print(colored('   [ V ] Trovato!', 'green'))
    device = Dobot(port=port)
    #homing(device)
    posizione_intermedia()
except:
    print(colored('  [ X ] Non ho trovato il Dobot', 'red'))

print('Cerco la camera...')
try:
    cap = cv2.VideoCapture(0)
    print(colored('   [ V ] Trovata!', 'green'))
except:
    print(colored('   [ X ] Non ho trovato la camera', 'red'))

print("Avvio gli algoritmi di IA...")
try:
    regressore_x = regression.crea_regressore_x()
    regressore_y = regression.crea_regressore_y()
    regressore_r = regression.crea_regressore_r()
    print(colored('   [ V ] Fatto!', 'green'))
except:
    print(colored('   [ X ] Ops, problema con gli algoritmi ', 'red'))
    
print('Ok, tutto fatto! Inizio a lavorare')
ferma_rullo(device)

cap.set(10, 70) # brightness     min: 0   , max: 255 , increment:1  
cap.set(11, 30) # contrast       min: 0   , max: 255 , increment:1     
cap.set(12, 30) # saturation     min: 0   , max: 255 , increment:1
cap.set(15, -5)
while True:
    lista_contorni =  []

    avanti_tappeto(device) # sposto la belt 
    time.sleep(0.4)
    
    _, frame = cap.read() # vedo l'immagine
    color_detection(frame) 
    
    guarda_e_prendi(frame, lista_contorni) # prendo gli oggetti
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# -------------- FUNZIONI UTILI --------------------------

    if cv2.waitKey(1) & 0xFF == ord('p'):
        pose = device.get_pose()
        position = pose.position
        stampa_posizione(device)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        pose = device.get_pose()
        position = pose.position
        device.move_to(position.x, position.y, position.z, position.r+10, True)
        
    if cv2.waitKey(1) & 0xFF == ord('b'):
        pose = device.get_pose()
        position = pose.position
        device.move_to(position.x, position.y, position.z, position.r-10, True)
     
    if cv2.waitKey(1) & 0xFF == ord('h'):
        print("Riposizioni il braccio...")
        homing(device)
        posizione_intermedia()
        print("   [ V ] Riposizionato!")
      
    if cv2.waitKey(1) & 0xFF == ord('k'):
        device.clear_alarms()  
        posizione_intermedia()

print(colored('>>> Termino l algoritmo <<<', 'blue'))
ferma_rullo(device)
cap.release()
cv2.destroyAllWindows()

   
