#Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

#Definir path para dataset
path = "Images"
files = os.listdir(path)

#lista de archivos en path
files.sort()

#imprimir para ver lista
print(files)

#Crear lista de imagenes y labels
image_array = []
label_array = []

#Ciclos a traves de cada archivo en files
for i in range(len(files)):
    sub_file = os.listdir(path+"/"+files[i])
    #print(len(sub_file))

    #Ciclo a traves de cada subcarpeta
    for j in range(len(sub_file)):
        #path para cada imagen
        # Ejemplo: Images/A/image_name1.jpg
        file_path = path+"/"+files[i]+"/"+sub_file[j]
        #leer cada imagen
        image = cv2.imread(file_path)
        #redimensionar imagen por 96x96
        image = cv2.resize(image, (96, 96))
        #convertir BRG imagen a RGB imagen
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Agregar esta imagen al array de imagenes
        image_array.append(image)
        #Agregar label al array
        label_array.append(i)

#Guardar y ejecutar para ver si funciona o no (Solo si tienes las imagenes listas)
#Despues de esto vuelve a ver el video en el minuto 15 para hacer lo siguiente:
#descomenta esto despues de ejecutar y hacer correcciones:

#convertir lista a array
image_array = np.array(image_array)
label_array = np.array(label_array,dtype="float")
#convertir el dataset en testeo y entrenamiento
from sklearn.model_selection import train_test_split

#salida
X_train, X_test, Y_train, Y_test = train_test_split(image_array, label_array, test_size=0.15)
#X_train contendra el 85% de las imagenes
#X_test contendra el 15% de las imagenes

#Elimiar arrays (Solo si vienes de la linea 134 cuando vas a iniciar a entrenar
#del image_array, label_array
#Esto para liberar memoria
#import gc
#gc.collect()


#Crear modelo
from keras import layers, callbacks, utils, applications, optimizers
from keras.models import Sequential, Model, load_model

model = Sequential()
#Agregar preentreno de los modelos a sequential model
pretrained_model = tf.keras.applications.EfficientNetB0(input_shape=(96, 96, 3), include_top=False)
model.add(pretrained_model)

#agregar pooling a model
model.add(layers.GlobalAveragePooling2D())

#agregar dropout a model (se agrega para aumentar el reconocimento)
model.add(layers.Dropout(0.3))

#Finalmente agregaremos dense layer como una salida
model.add(layers.Dense(1))

#Para algunas versiones de tensorflow necesitamos construir el modelo
model.build(input_shape=(None, 96, 96, 3))

#Para ver el model summary
model.summary()

#Guardar y ejecutar hasta aqui para ver el model sumary
#Asegurate de que tu pc este conectada a internet para descargar el preentrenamiento
#Tomara un poco de tiempo (si usas CPU tardara mucho mas tiempo)

#Despues de ejecutar el bloque anterior, ejecuta el siguiente
#Compilar modelos
model.complite(optimizer="adam", loss="mae", matrics=["mae"])

#Crear un checkpoint para guardar el mejor modelo
ckp_path = "trained_model/model"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckp_path,
    monitor="val_mae",
    mode="auto",
    save_best_only=True,
    save_weights_only=True
)

#create learning rate reducer to reduce lr when accuaracy does not improve
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.9,
    monitor="val_mae",
    mode="auto",
    cooldown=0,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

#start training model
Epochs = 100
Batch_Size = 32

#Select batch size acordado a tu tarjeta grafica
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    batch_size=Batch_Size,
    epochs=Epochs,
    callbacks=[model_checkpoint, reduce_lr]
)

#Antes de entrenar puedes elimiar image_array y label_array para incrementar memoria RAM

#Despues de entrenar el modelo
model.load_weights(ckp_path)
#Convertir modelo a tensorflow lite model
converter = tf.lite.TFLiteConverter.from_keras_model(Model)
tflite_model = converter.convert()

#save Model
with open("Model.tflite", "wb") as f:
    f.write(tflite_model)






