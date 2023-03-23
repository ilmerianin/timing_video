#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:39:20 2023

@author: wasilii
openCV
pip install cmake
!pip3 install face_recognition
!pip3 install dlib ????

"""
import cv2
import numpy as np
import pandas as pd
import os
import time
import ast
#import json
#from datetime import datetime
import face_recognition #https://github.com/ageitgey/face_recognition#face-recognition
import copy
#import matplotlib.pyplot as plt # Импортируем модуль pyplot библиотеки matplotlib для построения графиков
from openCVmet import * 

pathVideo = '/home/wasilii/Видео/try.mp4' # путь к видео

path = 'timingVideo.csv'

from PIL import Image, ImageDraw


verbose = False #True
#%% Рабочие библиотеки распознавания
def viewImageS(image,waiK= 500, nameWindow = 'message windows', verbose = True):
    ''' Вывод в отдельное окошко 
    image - картинка numpy подобный массив
    waiK - int время ожидания окна если 0- будет ждать нажатия клавиши
    nameWindow - название окна лучше по английски иначе проблемы с размером
    verbose - показывать или нет True/False
    '''
    if verbose:
        cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
        cv2.namedWindow('settings') # Окно настроек
        cv2.imshow(nameWindow, image)
        
        key = cv2.waitKey(waiK)
        if key ==27:
                print('Нажали клавишу 27')
        cv2.destroyAllWindows()
    else:
        pass
    return

def vievLandmark(image, face_landmarks_list):
    ''' Рисование губ глаз носа '''
# Load the jpg file into a numpy array
#image = face_recognition.load_image_file("biden.jpg")

# Find all facial features in all the faces in the image
#face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        #d = ImageDraw.Draw(pil_image, 'RGBA')
        d = ImageDraw.Draw(pil_image, mode ='RGB')
    
        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
    
        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
    
        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
    
        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
    
        pil_image.show()
        
        
    return 
def DravRectangleImage_face_rekogn(image, rectangle_NP, size_reduction_factor):
    ''' рисует картинку и квадраты фич работает 13.05.22
        image: cv.imread
        rectangle_NP :<class 'numpy.ndarray'> (x, 4) 
    return 
        True - 
        Fault - не найдены фичи
    '''
    k = int(1/size_reduction_factor)
    print('k:', k)
    #faces_detected = "Fich find: " + format(len(rectangle_NP))
    if len(rectangle_NP) == 0:
        print('не найдены фичи')
        return 0
    
    # image = np.ascontiguousarray(image, dtype=np.uint8)
    # Рисуем квадраты 
    for (y1, x2, y2, x1) in rectangle_NP:
         
        cv2.rectangle(image, (x1 *k, y1*k), (x2*k, y2*k), (255, 255, 0), 5) # отрисовка квадратов

    return image
def findEncodings(images):
    ''' кодирование лиц методом face_recognition кродирование списка лиц для распознания
        вход: набор images
        возврящает: list (эмбеддинг лиц?) лиц'''
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def Find_cascade_fich(face_cascade, image):
    ''' ищет лица на фото  методом openCV2 работает 13.05.22 cv.CascadeClassifier очень посредственно
    face_cascad 
    image frame 
    return:
        type(faces): <class 'numpy.ndarray'> (x, 4)
    время работы  wait time: 0.15 сек
     https://tproger.ru/translations/opencv-python-guide/'''

    assert not face_cascade.empty(), 'cv.CascadeClassifier( не нашёл файл haarcascade_frontalface_default.xml) '
    

    try:
        gray = cv2.cvtColor(image, cv.COLOR_BGR2GRAY)  # сделать серым
    except:
        gray = image

    faces = face_cascade.detectMultiScale(   #общая функция для распознавания как лиц, так и объектов. Чтобы функция искала именно лица, мы передаём ей соответствующий каскад.
        gray,              # Обрабатываемое изображение в градации серого.
        scaleFactor= 1.1,  # Параметр scaleFactor. Некоторые лица могут быть больше других, поскольку находятся ближе, чем остальные. Этот параметр компенсирует перспективу.
        minNeighbors= 5,   # Алгоритм распознавания использует скользящее окно во время распознавания объектов. Параметр minNeighbors определяет количество объектов вокруг лица. То есть чем больше значение этого параметра, тем больше аналогичных объектов необходимо алгоритму, чтобы он определил текущий объект, как лицо. Слишком маленькое значение увеличит количество ложных срабатываний, а слишком большое сделает алгоритм более требовательным.
        minSize=(10, 10)   # непосредственно размер этих областей
    )

    return faces

def findfith(image):
    ''' Подготовка класификатора OpenCV '''
                    # Собственно этой командой мы загружаем уже обученные классификаторы cv.data.haarcascades+'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = Find_cascade_fich(face_cascade,image) #  находит работает #поиск фитч "лиц"
    #print('Время распознания:', str(time.time()-StartTime), 'type:', type(faces), faces)
   
    return   faces   


def recognitionFacesCV(frame):
    '''  Распознание локаций лиц методом face_recognition'''        
    rectangle_NP = findfith(frame)
    if len(rectangle_NP) > 0:
        normRectangle = []
        for (x, y, w, h) in rectangle_NP:
            normRectangle.append([ y, x+w, y+h, x])
        return normRectangle
    return rectangle_NP

def recognitionFaces(frame):
    '''  Распознание локаций лиц методом face_recognition'''        
    return face_recognition.face_locations(frame)

def findFacesOnVideo(video_path, encodeListKnown =[], output = True, classNames = ['unknown'], faces_names = []):
    '''поиск лиц на вадео и запись их п пандас фрейм
    video_path :str путь к файлу,
    encodeListKnown  список известных лиц 
    output = True,  вывод в файл
    classNames = 'unknown'  - имена известных лиц
    , faces_names'''
    
    #face_recognition.face_landmarks(image)
    
    size_recovery_multiplier = 2
    size_reduction_factor = 0.25  # коэф уменьшения изобр Время работы: 4.513 min
                                  # коэф 1 уменьшения изобр Время работы: 43 min
    start_time = time.time()
    numFace = len(faces_names)
    
    df = pd.DataFrame({'frame': [0], 'name': [0], 'xyhw' : [0], 'encode': [0]})
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)  # загруз видео
                            #cv.CAP_DSHOW DirectShow (via videoInput)
                            #cv.CAP_FFMPEG Open and record video file or stream using the FFMPEG library.
                            # CAP_IMAGES  cv.CAP_IMAGES OpenCV Image Sequence (e.g. img_%02d.jpg)
    frame_wight = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if output: # формирование выходногопотока
        video_out_file = video_path.split('.')[-2] + '_out.mp4'
        outVid = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_wight,frame_height))
    
    streams = True
    num = 0
    blok = 10
    
    name = 'noName'
    while streams: # num < 50: #streams:
        
        ret ,fram =   cap.read() # захват кадра
        frame = copy.copy(fram)
        #frame = cv2.copyMakeBorder(fram,0,0,0,0,cv2.BORDER_REPLICATE)
        
        if not ret: # если кадр прочитан плохохо то берем следкющий
            blok -=1
            if blok ==0:
                streams = False
            continue
        #print(num, 'np.max(frame)', np.max(frame), '  np.average(frame)', round(np.average(frame)), frame.shape)
        frshape = frame.shape
        #viewImage(fram)
        frameRes = cv2.resize(frame.copy(), (0,0), None, size_reduction_factor, size_reduction_factor) # подготовка кадра
        
        #frameResBRGB = cv2.cvtColor(frameRes.copy(), cv2.COLOR_BGR2RGB)  
        frameResBRGB = frameRes[:, :, ::-1]     
        #frameResBRGB =   frameRes         
              # подготовка кадра
#        viewImage(frameResBRGB, waiK=0)
#        viewImage(frame, waiK=0)
        facesLocations = recognitionFacesCV(frameResBRGB)   # захват кадра
        encodingFaces = face_recognition.face_encodings(frameResBRGB, facesLocations) # кодирование лиц
        #landmark = face_recognition.face_landmarks(frameResBRGB)    # Поиск черт лица
        
# =============================================================================
#         print(f'кадр {num} количество bobox:', len(facesLocations), '\n facesLocations',type(facesLocations), facesLocations,
#               '\n encodingFaces:', type(encodingFaces), len(encodingFaces),'\n encodingFaces[0].shape'
#               '\n landmark:', type(landmark), len(landmark),
#               '\n encodeListKnown', type(encodeListKnown), len(encodeListKnown))
# =============================================================================
        print('\r', num, ' фитчи ',len(facesLocations), end=' ')
#        viewImage(DravRectangleImage_face_rekogn(frame.copy(), facesLocations, size_reduction_factor), waiK=500)

        #viewImage(fram)
        if len(facesLocations) > 0:
            box =len(facesLocations)
               # Поверка на новые лица
               # синхронный перебор кодов лиц и локаций
            for encodingFace, faseLoc in zip(encodingFaces, facesLocations, ):
                
                # vievLandmark(frame, landmark) #подрисовка черт
                #viewImage(DravRectangleImage_face_rekogn(frame, facesLocations, size_reduction_factor),nameWindow=f'frame {num} faces {box} shape: {frshape}')

                #viewImage(DravRectangleImage(frame, landmark), waiK=0, nameWindow='landmark')
                #faseDict = {}

                y1, x2, y2, x1 = faseLoc # преобразование координат
                faseLoc = (  int(x1/size_reduction_factor) ,int(y1/size_reduction_factor), int(x2/size_reduction_factor), int(y2/size_reduction_factor))
                x1, y1, x2, y2 = faseLoc
                #y1, x2, y2, x1 = y1 * size_recovery_multiplier, x2 * size_recovery_multiplier, y2 * size_recovery_multiplier, x1 * size_recovery_multiplier
                   # поиск знакомых лиц
                if len(encodeListKnown) > 0:
                                             # поиск  в знакомых код лицах    кода лицаа
                    matches = face_recognition.compare_faces(encodeListKnown, encodingFace) # лица муж?
                    faceDist = face_recognition.face_distance(encodeListKnown, encodingFace)   # расстояние
                    
                    minFaseIdInd = np.argmin(faceDist) # самое ближнее лицо маска True/False
                                                        #        [True]      [0.43499996]
                    print(f'кадр {num} Расстояние matches: ', matches, 'faceDist:', faceDist)
                    
                    if faceDist[minFaseIdInd] < 0.58: # по моему бред просто выбор ближнего
                        name = faces_names[minFaseIdInd]
                        #viewImage(frame[y1:y2, x1:x2,...], waiK=0)
                    else: # если новая фитча
                        #print('shape:', frame.shape, frame[y1:y2, x1:x2,...].shape)
                        viewImage(frame[y1:y2, x1:x2,...], waiK=0,verbose=verbose, nameWindow = '2 video')
                        numFace +=1
                        name = str(numFace)
                        #name = str(input(f'Введите Имя клиента пака похож на {faces_names[minFaseIdInd]}>>>>>>'))

                else:   # первая новая фитча
                        viewImage(frame[y1:y2, x1:x2,...],waiK=0, verbose=verbose, nameWindow = '1 video')
                        numFace +=1
                        name = str(numFace)
                        #name = str(input('Введите Имя клиента>>>>>>'))
                    
                print('name:', name, type(encodingFace) )
                
                dfN = pd.DataFrame(columns = ['frame', 'name', 'xyhw', 'encode'])
                dfN['encode'] = dfN['encode'].astype(object)
                dfN['xyhw'] = dfN['xyhw'].astype(object)
                dfN['encode'] = [np.array(encodingFace),]
                dfN['name'] = str(name)
                dfN['frame'] = num
                dfN['xyhw'] = [faseLoc]
                #print(dfN)

                df = pd.concat([df ,dfN])
                #print('df.shape', df.shape)
                                # Занесение новых лиу в базу
                if name not in  faces_names: 
                    faces_names.append(name)
                    encodeListKnown.append(encodingFace)
                    classNames.append(name)
                    
                # Вывод боксов с именем
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3) # отрисовка квадратов )
                cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,0, 255), cv2.FILLED) # минмибокс для текста
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, str(name), (x1, y2-6), font, 1.0, (0, 0, 0), 1) # вывод текста

                viewImage(frame,waiK= 0, verbose=verbose, nameWindow = 'video')
                                   
        if output:
            #cv2.imshow('img RGB', frame)
            outVid.write(frame)
            #viewImage(frame, waiK=800)
            pass
        num +=1 # счетчик кадров

    df.to_csv('timingVideo.csv',index=False)
    print('faces_names', len(faces_names),faces_names )
    print('encodeListKnown', len(encodeListKnown))
    print('df.shape', df.shape)
    print('Время работы:', (time.time()-start_time)/60, 'min')
#%% Чтение и подготовка df


def from_np_array(array_string):
    ''' преобразоваие в np тз строчки pd '''
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))
def from_np_array1(array_string):
    ''' преобразоваие в np тз строчки pd '''
    array_string = ''.join(array_string.replace('( ', ')').split())
    return np.array(ast.literal_eval(array_string))


def wievFrame_getName(faceLoc, frameNum, pathVideo): #вывод лица и ввод имени
    ''' Вывод на экран лиц из списка и ввод их имен'''
    print('faceLoc', faceLoc, len(faceLoc), type(faceLoc) )
    x1, y1, x2, y2= faceLoc
    print(x1, y1, x2, y2, type(x1))
    cap = cv2.VideoCapture(pathVideo)
#    for fra in range(frameNum):
#        _, frameIm = cap.read()
#        pass
    cap.set(cv2.CAP_PROP_POS_FRAMES,frameNum ) # выставить в нужный frame
    _, frameIm = cap.read()
    print(frameIm.shape)
    
    viewImage(frameIm[y1:y2, x1:x2,...], waiK=0, nameWindow = 'Face')
    cap.release() # отпустить поток возможно не нужно для файлов
    return str(input(' Введите название фитчи>> '))

def from_df_face_name(path, pathVideo):
    ''' считывание из .csv имен и кодов лиц перобразование прочитанных  np.array из строчки в реальный np.appay 
    вывод на экран
    возврат:
        faces_names, имя классов
        encodeListKnown вектор вица (face recognition)
        '''
    df = pd.read_csv(path, converters = {'encode':from_np_array,'xyhw':from_np_array1 })
    faces_names = []
    encodeListKnown = []
    
    listName = df['name'].unique().tolist() 
    
    for name in listName:
        name_df = df[df['name'] == name]# ['encode'].values
        
        faceLoc = name_df['xyhw'].values  
        frame = name_df['frame'].values
        encode = name_df['encode'].values
        print( type(encode),encode.shape[0])        
        if encode.shape[0] > 2:
            print(name, faceLoc)
            name = wievFrame_getName(faceLoc[0],frame[0], pathVideo) #вывод лица и ввод имени
            
            
            faces_names.append(str(name))
            encodeListKnown.append(encode[0])
    return faces_names, encodeListKnown

#%% KMeans
from sklearn.cluster import KMeans # Импортируем библиотеки KMeans для кластеризации
def klastMean(embedding):
    '''  метод для поиска центра класса векторов одного лица не проверен'''
    # n_clusters = 6        # максимальное  количество кластеров
    cost = []                     # контейнер под список
    kmean = KMeans(1)           # Создаем объект KMeans с i-классами
    kmean.fit(embedding)      # Проводим классетризацию 
    centers = kmean.cluster_centers_
    
    print('центр кластера', centers.shape)
    return centers[0]

#%%

#embedding = np.concatenate(items_meta.iloc[:]['embeddings'].values).reshape((-1,312))


def normalisClassesEncod(path):
    '''  считывание из .csv имен и кодов лиц перобразование прочитанных  np.array из строчки в реальный np.appay 
    поиск центра занчения класса  через kmeans для кодов лица'''
    faces_names = []
    encodeListKnown = []
    
    df = pd.read_csv(path, converters = {'encode':from_np_array,'xyhw':from_np_array1 })
    classes = df['name'].unique()
    print(classes)
    for oneClas in classes:
        faces_names.append(str(oneClas))
        
        df_oneClass = df[df['name'] == oneClas]['encode'].values
        print(df_oneClass)
        if df_oneClass != 0:
            encodeListKnown.append( klastMean(np.concatenate(df_oneClass).reshape(-1, df_oneClass[0].shape[0])) ) 
    return faces_names, encodeListKnown

def from_df_to_nameEncod(path):
    ''' считывание из .csv имен и кодов лиц перобразование прочитанных  np.array из строчки в реальный np.appay
    берется первый из кодов лица'''
    df = pd.read_csv(path, converters = {'encode':from_np_array})
    faces_names = []
    encodeListKnown = []
    
    listName = df['name'].unique().tolist() 
    for name in listName:
        encode = df[df['name'] == name]['encode'].values
        print( type(encode),encode.shape[0])        
        if encode.shape[0] > 2:
            faces_names.append(str(name))
            encodeListKnown.append(encode[0])
    return faces_names, encodeListKnown
    

                
def renameClacess(faces_names):
    '''  Переименование загруженных классов 
    перебирает по очереди и просит ввести новое если ентер то конец ввода'''
    for i in range(len(faces_names)):
        print('class:', faces_names[i])
        inp  = str(input(f' Class : {faces_names[i]} new name or enter>> '))
        if inp !='':
            faces_names[i] = inp
        else:
            break
        
    return faces_names


def main(rename = True, loadClasses = True): # True  False
    ''' Пооход по видео и сохранение найденных фитч в пандас потом вывод и запрос имен 
    rename: - меню переименивание загруженных классов
    
    loadClasses - Загрузка сохраненных кодов лиц
    rename = True задание имен классов
    
    '''
    encodeListKnown =[]
    faces_names = []
    
    print('go')
    if loadClasses:
        print('Загрузка сохраненных кодов лиц ВКЛЮЧЕННА')
        if rename:
            print('Переименивание загруженных классов включено')
        else:
            print('Переименивание загруженных классов ВЫКЛЮЧЕНО')
    else:
        print('Базы по лицам НЕТ')

    if loadClasses: #агрузка сохраненных кодов лиц ВКЛЮЧЕННА'
        faces_names, encodeListKnown = normalisClassesEncod(path)
        
        if rename: #Переименивание загруженных классов
            #faces_names = renameClacess(faces_names)  
            faces_names, encodeListKnown= from_df_face_name(path, pathVideo) # вариант с просмотром

        print('найденны имена:', faces_names,len(faces_names))
        print('найденны encod', type(encodeListKnown),len(encodeListKnown), encodeListKnown[0].shape)
    
        
    findFacesOnVideo(pathVideo,encodeListKnown =encodeListKnown ,faces_names=faces_names)
          
                
if __name__=='__main__':
    main()
    
    
  # конфигурация лица  
  # использовалась для проверки работоы алгоритмаы
landmark1 = [{'chin': [(542, 217), (539, 225), (538, 233), (538, 242), (538, 251), (539, 261), (541, 270), (544, 279), (553, 284), (566, 286), (583, 283), (600, 279), (614, 272), (623, 260), (627, 247), (629, 233), (632, 218)],
              'left_eyebrow': [(532, 199), (535, 198), (539, 201), (543, 206), (547, 211)], 
              'right_eyebrow': [(560, 212), (569, 209), (579, 207), (590, 210), (600, 214)], 
              'nose_bridge': [(552, 219), (548, 225), (545, 232), (540, 239)], 
              'nose_tip': [(542, 244), (543, 246), (545, 247), (550, 247), (554, 248)], 
              'left_eye': [(537, 215), (540, 214), (545, 215), (550, 219), (544, 220), (539, 219)], 
              'right_eye': [(569, 222), (574, 220), (581, 221), (588, 223), (581, 226), (574, 225)], 
              'top_lip': [(543, 256), (542, 255), (544, 255), (546, 256), (549, 256), (558, 259), (569, 261), (565, 261), (549, 260), (546, 259), (544, 258), (545, 257)],
              'bottom_lip': [(569, 261), (558, 263), (550, 263), (547, 263), (544, 261), (543, 259), (543, 256), (545, 257), (545, 256), (547, 257), (550, 258), (565, 261)]}]  
    
  

#encodeListKnown 9
#df.shape (217, 4)
#Время работы: 3.211605203151703 min
# применение 
    # frameResBRGB = frameRes[:, :, ::-1]  
    # encodeListKnown 8
    # df.shape (218, 4)
    # Время работы: 3.064599951108297 min
# применение    
    # frameResBRGB = cv2.cvtColor(frameRes.copy(), cv2.COLOR_BGR2RGB) 
    # encodeListKnown 8
    # df.shape (218, 4)
    # Время работы: 3.09690744082133 min
# применение поиска лиц openCV: по фитчам найденным ранее
    # encodeListKnown 8
    # df.shape (107, 4)
    # Время работы: 2.6513411164283753 min
#  применение поиска лиц openCV: по фитчам  пустому списку
    # encodeListKnown 9
    # df.shape (107, 4)
    # Время работы: 2.5944750905036926 min
    
    
    
# =============================================================================
#  faces_names1 = ['Vasilii',
#           'tatu',
#           'hand',
#           'tatu 56',
#           'Passenger очки',
#           'clothes',
#           'Passenger  ',
#           'Passenger  60']   
# =============================================================================
    
    
    
    
    
    
    
    