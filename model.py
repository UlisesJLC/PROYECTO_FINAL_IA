from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import cv2
from pathlib import Path

paths=["train_covid", "train_normal"]
imgs=[]
labels=[]
for idx,path in enumerate(paths):
    images_path = Path(path).glob("**/*.jpg")
    images_path = [str(x) for x in images_path]
    for image_path in images_path:
        img =(cv2.imread(image_path,0))
        img = img/255
        img = cv2.resize(img,(150,150),cv2.INTER_AREA)
        imgs.append(img)
        labels.append(idx)
imgs=np.array(imgs)
labels=np.array(labels)
print(f"images shape: {imgs.shape}")
print(f"Num of labels: {len(labels)}")
"""
imgs_test=[]
img=cv2.imread("C:\\xampp\\htdocs\\IA\\uploads\\imagen.jpg", 1)
img=img/255
imgs_test.append(cv2.resize(img,(150,150),cv2.INTER_AREA))     
imgs_test=np.array(imgs_test)
print(f"images shape: {imgs_test.shape}")
"""
images = np.expand_dims(imgs,axis=-1)
model = keras.Sequential([
    keras.layers.Input(shape = (150,150,1)),
    #keras.layers.Rescaling(1/255), #Es para no hacer lo del for 
    keras.layers.Conv2D(32,3,strides=1,padding="same",activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64,3,strides=1,padding="same",activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128,3,strides=1,padding="same",activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(115, activation="softmax")
])
print(model.summary())


model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])
history = model.fit(
    imgs,
    labels,
    epochs=30,
    batch_size=32, 
    validation_split=0.2)

model.save('mi_modelo.h5') 
