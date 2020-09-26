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

    autoencoder.load_weights('C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\faces_upscale_model_color')  

    return autoencoder.predict(np.array([img]))[0]













pygame.init()


display_width = 600
display_height = 600

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Enhance')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()




x =  (display_width * 0.45)
y = (display_height * 0.8)

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

    #path = "C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\20200407_112346.jpg"
    path = "C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\82536cacddfc4394831966753f996d9f.jpeg"
    #path = "C:\\Users\\lmgab\Downloads\\thumbnails128x128-20200624T204854Z-001\\thumbnails128x128\\00000\\00000.png"
    img = cv2.imread(path,1)
    originalFace = pygame.image.load(path)
    orignalRes = originalFace.get_size()
    originalFace = pygame.transform.scale(originalFace,[300,300])
    gameDisplay.blit(originalFace, (0,0))
    
    if(pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0]<300 and pygame.mouse.get_pos()[1]<300):
        if(mouseC == 0):
            startPos = pygame.mouse.get_pos()
            pygame.mouse.get_rel()
            selected = False
        if(mouseC > 0):
            pygame.draw.rect(gameDisplay,(25,25,25),[startPos[0],startPos[1],pygame.mouse.get_pos()[1]-startPos[1],pygame.mouse.get_pos()[1]-startPos[1]],3)
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


        for y in img[start_new_y:+start_new_y+end_new]:#img[int(round(startPos[0]/(300/len(img)))):int(round(startPos[0]/(300/len(img))+endPos[0]/(300/len(img))))]:
            temp2 = []
            for x in y[start_new_x:start_new_x+end_new]:#y[int(round(startPos[1]/(300/len(img)))):int(round(startPos[1]/(300/len(img))+endPos[0]/(300/len(img))))]:
                temp2.append(x)
            temp.append(np.array(temp2))
        crop_img = np.array([np.array(temp)])/255
        


        
        
        #print(crop_img.shape)
        new = enhance(crop_img[0])
        #print(new.shape)

        
        # ax = plt.subplot(2, 1, 1)
        # plt.imshow(crop_img.reshape(len(crop_img[0]), len(crop_img[0]), 3))
        # ax = plt.subplot(2, 1, 2)
        # plt.imshow(new.reshape(len(new), len(new), 3))
        # plt.show()


        new_output = np.array([new])*255
        cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg", new_output[0]) 
        new_output2 = crop_img*255
        cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic2.jpg", new_output2[0]) 
        
        selected = True
        mouseC = 0

    if(selected):
        pygame.draw.rect(gameDisplay,(25,25,25),[startPos[0],startPos[1],endPos[0],endPos[0]],3)

        #x = gameDisplay.blit(pygame.transform.scale(originalFace,[int(300*ratio),int(300*ratio)]), (0, 300),cropRect)

        smallFace = pygame.image.load("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic2.jpg")
        smallFace = pygame.transform.scale(smallFace,[300,300])
        gameDisplay.blit(smallFace, (0,300))

        newFace = pygame.image.load("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg")
        newFace = pygame.transform.scale(newFace,[300,300])
        gameDisplay.blit(newFace, (300,300))


        if(pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0]>300 and pygame.mouse.get_pos()[1]<300):
            img = cv2.imread("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg",1)
            imgs = np.array([np.array(img)])/255
            new = enhance(imgs[0])
            new_output = np.array([new])*255
            cv2.imwrite("C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\tempPic.jpg", new_output[0]) 


    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()