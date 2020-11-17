import cv2
import numpy as np

def prova_prendere(x,y):
    posizione_intermedia()
    apri_chela(device)
    r=10.0
    device.move_to(x-5, y+10.0, 5.0, r, True) 
    for n in range(5):
        device.move_to(x-10, y+10.0, -3.0, r, True)
        chiudi_chela(device)
        pose = device.get_pose()
        position = pose.position
        r+=10.0
        n+=1
        device.move_to(x-10, y+10.0, 5.0, r, True)
        apri_chela(device)
        
def esaspera_colori(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rows,cols,_ = frame.shape  
    for x in range(rows):
        for y in range(cols): 
            pixel = frame[x,y]           
            hue = pixel[0]
            saturation = pixel[1]
            lightness = pixel[2]

            frame[x,y][1] = frame[x,y][1] + (frame[x,y][1]/2)   
            frame[x,y][2] = frame[x,y][2] + (frame[x,y][2]/2)
            
            # check per superamento limite di 255
            if frame[x,y][1] > 255:
                frame[x,y][1] = 255
            if frame[x,y][2] > 255:
                frame[x,y][2] = 255
                          
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    return frame

def cose():
    
    # calcolo l'orientamento
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    left_x, left_y = leftmost
    right_x, right_y = rightmost
    top_x, top_y = topmost
    bottom_x, bottom_y = bottommost 
    
    #centro
    #  cv2.circle(frame, (int(x_centro),int(y_centro)), 2,(0, 255, 255), 2) # centro
    #  cv2.putText(frame, str('x_centro: '+str(x_centro)+'| y_centro: '+ str(y_centro)), (int(x_centro-15),int(y_centro-15)), 1, 1, (255, 255, 255)) # scrivo la forma      
    #sx
    #  cv2.circle(frame, (int(left_x), int(left_y)), 2,(255, 255, 255), 2) # sx
    #  cv2.putText(frame, str('x_sx: '+str(left_x)+'| y_sx: '+ str(left_y)), (int(left_x-15), int(left_y-15)), 1, 1, (255, 255, 255)) # scrivo la forma    
    #dx
    #  cv2.circle(frame, (int(right_x), int(right_y)), 2,(255, 255, 255), 2) # dx
    #  cv2.putText(frame, str('x_dx: '+str(right_x)+'| y_dx: '+ str(right_y)), (int(right_x-15), int(right_y-15)), 1, 1, (255, 255, 255)) # scrivo la forma       
    #top
    #  cv2.circle(frame, (int(top_x), int(top_y)), 2,(255, 255, 255), 2) # top
    #  cv2.putText(frame, str('x_top: '+str(top_x)+'| y_top: '+ str(top_y)), (int(top_x-15), int(top_y-15)), 1, 1, (255, 255, 255)) # scrivo la forma
    #bottom
    #  cv2.circle(frame, (int(bottom_x), int(bottom_y)), 2,(255, 255, 255), 2) # top
    #  cv2.putText(frame, str('x_bottom: '+str(bottom_x)+'| y_bottom: '+ str(bottom_y)),(int(bottom_x-15), int(bottom_y-15)), 1, 1, (255, 255, 255)) # scrivo la form

def stampa_pixel():
    print('x dell oggetto in pixel ', (x_centro))
    print('y dell oggetto in pixel ', (y_centro))
    pose = device.get_pose()
    position = pose.position
    print('Dobot', position)
    
def color_detection(frame):
    #cv2.imshow('originale', frame)
    frame_rosso = frame.copy()
    frame_rosa = frame.copy()
    frame_arancione = frame.copy()
    frame_blu = frame.copy()
    frame_verde = frame.copy()
    frame_giallo = frame.copy()
    
    frame_colori_selezionati = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #--------------------- COLORE ARANCIONE -------------------------------------------
    low_arancione = np.array(  [10,100,0])
    high_arancione = np.array(  [18,255,255])
    arancione_mask = cv2.inRange(frame, low_arancione, high_arancione)
    
    solo_arancione = cv2.bitwise_and(frame_arancione, frame_arancione, mask=arancione_mask)
    
    #--------------------- COLORE ROSA -------------------------------------------
    low_rosa = np.array(  [0,0,70])
    high_rosa = np.array(  [60,80,255])
    rosa_mask = cv2.inRange(frame, low_rosa, high_rosa)
    
    solo_rosa = cv2.bitwise_and(frame_rosa, frame_rosa, mask=rosa_mask)
    
    # ------------------- COLORE ROSSO --------------------------------------------
    lower_red = np.array(  [0,120,70])
    upper_red = np.array(  [10,255,255])
    rosso_mask1= cv2.inRange(frame, lower_red, upper_red)

    lower_red = np.array(  [170,120,70])
    upper_red = np.array(  [180,255,255])
    rosso_mask2 = cv2.inRange(frame,lower_red,upper_red)
    # maschera completa
    rosso_mask = rosso_mask1 + rosso_mask2
    solo_rosso = cv2.bitwise_and(frame_rosso, frame_rosso, mask=rosso_mask)

    # ------------------- COLORE BLU --------------------------------------------
    # colore blu
    low_blu = np.array(  [90, 50, 50])
    high_blu = np.array(  [126, 255, 255])
    blu_mask = cv2.inRange(frame, low_blu, high_blu)
    
    solo_blu = cv2.bitwise_and(frame_blu, frame_blu, mask=blu_mask)

    # ------------------- COLORE VERDE --------------------------------------------
    # colore verde
    low_verde = np.array(  [40, 50, 50])
    high_verde = np.array(  [80, 255, 255])
    verde_mask = cv2.inRange(frame, low_verde, high_verde)
    
    solo_verde = cv2.bitwise_and(frame_verde, frame_verde, mask=verde_mask)

    # ------------------- COLORE GIALLO --------------------------------------------
    #colore giallo

    low_giallo = np.array(  [18, 50, 50])
    high_giallo = np.array(  [32, 255, 255])
    giallo_mask = cv2.inRange(frame, low_giallo, high_giallo)
    
    solo_giallo = cv2.bitwise_and(frame_giallo, frame_giallo, mask=giallo_mask)


    # ------------------- STAMPO E RITORNO --------------------------------------------
    #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    #cv2.imshow('rosso', solo_rosso)
    #cv2.imshow('blu', solo_blu)
    #cv2.imshow('verde', solo_verde)
    #cv2.imshow('giallo', solo_giallo)
    #cv2.imshow('rosa', solo_rosa)
    #cv2.imshow('arancione', solo_arancione)
    
    mask_totale = arancione_mask + rosso_mask + giallo_mask + verde_mask + blu_mask 
    solo_colori = cv2.bitwise_and(frame_colori_selezionati, frame_colori_selezionati, mask=mask_totale)
    cv2.imshow('colori', solo_colori)
    
    return solo_rosso,solo_blu,solo_verde,solo_giallo,solo_rosa,solo_arancione

def crea_regressore_z():
    train_for_x = area_perimetro_oggetti.iloc[0:]
    label_for_x = altezza_dobot.iloc[0:]
    #print(label_for_x)
    #print(train_for_x)

    degree=2
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(train_for_x, label_for_x)
    
    model_Z = ElasticNetCV()
    model_Z.fit(train_for_x, label_for_x)
    
    #r_sq = model_Z.score(train_for_x, label_for_x)
    #print('coefficient of determination Z:', r_sq)
    return model_Z

def z_predizione(model, val1, val2):
    lista = []
    lista = lista + [val1, val2]
    with open('test.csv', 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, escapechar='', quotechar=' ')
        wr.writerow(lista)
    test = pd.read_csv('test.csv', header=None)
    
    # ASSISTENTE
    train_for_x = area_perimetro_oggetti.iloc[0:]
    label_for_x = altezza_dobot.iloc[0:]
    degree=1
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(train_for_x, label_for_x)
    
    predizione = 	(polyreg.predict(test) + model.predict(test))/2
    print('Z', predizione)

    return predizione
    
# MAIN
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_SETTINGS, 1)

while True: 
    _, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    
    