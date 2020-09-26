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


samples = 20000
testPercent = .1
epochs = 1
batchSize = 128

x_train = []
y_train = []
c = 0
path = "----FLICKR FACES DATASET----"
folders = os.listdir(path)
files = []
pics = []
for f in folders:
    for p in os.listdir(path+"\\"+f):
        img = cv2.imread(path+"\\"+f+"\\"+p,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_train.append(cv2.resize(img,(32, 32)))
        y_train.append(cv2.resize(img,(64, 64)))
        c+=1
        if(c==samples):
            break
        print("Importing Image: "+str(c+1),end='\r')
    if(c==samples):
        break
print("\n")

x_train = np.array(x_train).reshape(samples, 32, 32, 3)/255
y_train = np.array(y_train).reshape(samples, 64, 64, 3)/255



x_test = np.array(x_train[int(round(samples*(1-testPercent))):])
y_test = np.array(y_train[int(round(samples*(1-testPercent))):])
x_train = np.array(x_train[:int(round(samples*(1-testPercent)))])
y_train = np.array(y_train[:int(round(samples*(1-testPercent)))])


# plt.imshow(y_train[0].reshape(64, 64, 3))
# plt.show()



input_img = Input(shape=(32, 32, 3))  # adapt this if using `channels_first` image data format
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
print(autoencoder.summary())

#autoencoder.load_weights('C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\faces_upscale_model_color')
autoencoder.fit(x_train, y_train,
                epochs = epochs,
                batch_size = batchSize,
                shuffle = True,
                validation_data = (x_test, y_test)
                )
autoencoder.save('C:\\Users\\lmgab\\Desktop\\mainEnv\\faces\\faces_upscale_model_color')





decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(y_test[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
