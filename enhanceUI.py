from tensorflow.keras.layers import Input, Dense, Conv2D, UpSampling2D, UpSampling3D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import numpy as np
import cv2
import os
import pygame
from tkinter.filedialog import askopenfilename 
from tkinter import *

root = Tk()
foo = askopenfilename()
root.destroy()

def enhance(img):
    #print(len(img))
    input_img = Input(shape=(len(img), len(img[0]), 3))  # adapt this if using `channels_first` image data format
    x = UpSampling3D((2, 2, 1))(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    opt = optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam"
    )
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')

    autoencoder.load_weights('C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\faces_upscale_model_color_4')  

    return autoencoder.predict(np.array([img]))[0]

pygame.init()
pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
myfont = pygame.font.SysFont('Arial', 30)
myfont2 = pygame.font.SysFont('Arial', 20)

display_width = 600
display_height = 600

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Enhance')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()

crashed = False
mouseC = 0
selected = False
startPos = (0,0)
endpos = (0,0)
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    gameDisplay.fill(white)

    path = foo
    #path = "C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\taipei-concert-coronavirus-eric-chou-anrong-xu-006.jpg"
    #path = "C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\20200407_112346.jpg"
    #path = "C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\82536cacddfc4394831966753f996d9f.jpeg"
    #path = "C:\\Users\\lmgab\Downloads\\thumbnails128x128-20200624T204854Z-001\\thumbnails128x128\\00000\\00000.png"
    img = cv2.imread(path,1)
    originalFace = pygame.image.load(path)
    orignalRes = originalFace.get_size()
    originalFace = pygame.transform.scale(originalFace,[300,300])
    gameDisplay.blit(originalFace, (0,0))

    pygame.draw.rect(gameDisplay,(200,200,200),[300,0,300,300])
    pygame.draw.line(gameDisplay,(25,25,25),(300,150),(600,150),5)
    pygame.draw.line(gameDisplay,(25,25,25),(450,0),(450,300),5)
    pygame.draw.line(gameDisplay,(25,25,25),(450,75),(600,75),5)
    textsurface = myfont.render('Enhance', False, (0, 0, 0))
    gameDisplay.blit(textsurface,(328,60))
    textsurface = myfont.render('Zoom In', False, (0, 0, 0))
    gameDisplay.blit(textsurface,(480,20))
    textsurface = myfont.render('Zoom Out', False, (0, 0, 0))
    gameDisplay.blit(textsurface,(473,88))
    textsurface = myfont.render('New Image', False, (0, 0, 0))
    gameDisplay.blit(textsurface,(312,200))

    if(pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0]<300 and pygame.mouse.get_pos()[1]<300):
        if(mouseC == 0):
            startPos = pygame.mouse.get_pos()
            pygame.mouse.get_rel()
            selected = False
        if(mouseC > 0):
            pygame.draw.rect(gameDisplay,(255,25,25),[startPos[0],startPos[1],pygame.mouse.get_pos()[1]-startPos[1],pygame.mouse.get_pos()[1]-startPos[1]],3)
        mouseC += 1

    if(mouseC>0 and pygame.mouse.get_pressed()[0] == False and pygame.mouse.get_pos()[0]<300 and pygame.mouse.get_pos()[1]<300):
        endPos = (pygame.mouse.get_rel()[1],pygame.mouse.get_rel()[1])
        ratio = 300/endPos[0]
        cropRect = (int(startPos[0]*ratio),int(startPos[1]*ratio),int(endPos[0]*ratio),int(endPos[0]*ratio))

        temp = []

        start_old_x = startPos[0]
        start_old_y = startPos[1]
        start_new_x = int(start_old_x/(300/len(img)))
        start_new_y = int(start_old_y/(300/len(img)))
        end_new = int(endPos[0]/(300/len(img)))

        for y in img[start_new_y:+start_new_y+end_new]:
            temp2 = []
            for x in y[start_new_x:start_new_x+end_new]:
                temp2.append(x)
            temp.append(np.array(temp2))
        crop_img = np.array([np.array(temp)])/255
        
        new = enhance(crop_img[0])
        new_output = np.array([new])*255
        cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg", new_output[0]) 
        new_output2 = crop_img*255
        cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic2.jpg", new_output2[0]) 
        
        selected = True
        mouseC = 0
        newC = 300

    if(selected):
        pygame.draw.rect(gameDisplay,(255,25,25),[startPos[0],startPos[1],endPos[0],endPos[0]],3)

        smallFace = pygame.image.load("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic2.jpg")
        smallFace = pygame.transform.scale(smallFace,[300,300])
        gameDisplay.blit(smallFace, (0,300))

        newFace = pygame.image.load("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg")
        imTemp = cv2.imread
        newFace = pygame.transform.scale(newFace,[300,300])
        gameDisplay.blit(newFace, (300,300))
        img = cv2.imread("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg",1)
        textsurface = myfont2.render(str(len(img))+"x"+str(len(img)), False, (255, 1, 1))
        gameDisplay.blit(textsurface,(305,300))

        if(pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0]>450 and pygame.mouse.get_pos()[1]<150):
            img = cv2.imread(path,1)
            if(pygame.mouse.get_pos()[1]<75 and len(img)-start_new_x > 20):
                start_new_x +=2
                start_new_y +=2
                end_new -=4
            if(pygame.mouse.get_pos()[1]<150 and pygame.mouse.get_pos()[1]>75 and start_new_x+20 < len(img)):
                start_new_x -=2
                start_new_y -=2
                end_new +=4
            temp = []
            for y in img[start_new_y:+start_new_y+end_new]:
                temp2 = []
                for x in y[start_new_x:start_new_x+end_new]:
                    temp2.append(x)
                temp.append(np.array(temp2))
            crop_img = np.array([np.array(temp)])/255
            new_output2 = crop_img*255
            cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg", new_output2[0]) 
            cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic2.jpg", new_output2[0]) 
            selected = True
            mouseC = 0





        if(pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0]>450 and pygame.mouse.get_pos()[1]<300  and pygame.mouse.get_pos()[1]>150):
            img = cv2.imread(path,1)
            if(pygame.mouse.get_pos()[0]<450+37):
                start_new_x -=2
            if(pygame.mouse.get_pos()[0]>450+113):
                start_new_x +=2
            if(pygame.mouse.get_pos()[1]<150+37):
                start_new_y -=2
            if(pygame.mouse.get_pos()[1]>150+113):
                start_new_y +=2
            temp = []
            for y in img[start_new_y:+start_new_y+end_new]:
                temp2 = []
                for x in y[start_new_x:start_new_x+end_new]:
                    temp2.append(x)
                temp.append(np.array(temp2))
            crop_img = np.array([np.array(temp)])/255
            new_output2 = crop_img*255
            cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg", new_output2[0]) 
            cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic2.jpg", new_output2[0]) 
            selected = True
            mouseC = 0






        if(pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0]>300 and pygame.mouse.get_pos()[0]<450 and pygame.mouse.get_pos()[1]<150):
            img = cv2.imread("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg",1)
            imgs = np.array([np.array(img)])/255
            new = enhance(imgs[0])
            new_output = np.array([new])*255
            cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg", new_output[0]) 
            newC=300


        if(newC<600):
            pygame.draw.rect(gameDisplay,(255,255,255),[300,newC,300,300],0)
            newC+=10

        if(pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0]>300 and pygame.mouse.get_pos()[0]<450 and pygame.mouse.get_pos()[1]>150 and pygame.mouse.get_pos()[1]<300):
            root = Tk()
            foo = askopenfilename()
            root.destroy()
            selected = False

        
    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
