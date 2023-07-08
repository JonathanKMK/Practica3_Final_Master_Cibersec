# Practica3_Final_Master_Cibersec
Practica Final 3 - Master Ciberseguridad UIDE-EIG

# Librerias utilizdas 
Tensorflow: Para el aprendizaje automático y la construcción de redes neuronales.
Numpy: Proporciona estructuras de datos y funciones para trabajar con matrices.
OS: Proporciona funciones para interactuar con el sistema operativo, como acceder a rutas de archivos y directorios.
Matplotlib: Se utiliza para visualizar datos en forma de gráficos y diagramas.

# Moldulo Importados
Sequential: Se utiliza para construir modelos de redes neuronales secuenciales capa por capa.
Layers: Contiene varias capas predefinidas que se pueden utilizar para construir modelos de redes neuronales.
Keras: Proporciona una interfaz de alto nivel para construir y entrenar modelos de aprendizaje automático.
Image: Proporciona herramientas para el preprocesamiento de imágenes, como cargar y transformar imágenes.
confusion_matrix: Se utiliza para calcular la matriz de confusión.
train_test_split: Se utiliza para dividir conjuntos de datos en conjuntos de entrenamiento y prueba.
LabelEncoder: Se utiliza para codificar etiquetas de clase en valores numéricos.

# Se procede a guardar la ruta de los directorios de train y test en variables
directorio_train = "C:/Users/jonat/Desktop/CarneDataset/train"
directorio_test = "C:/Users/jonat/Desktop/CarneDataset/test"

# Con las libreras y modulos importados se procede a realizar la lectura de las imagenes tanto de train y test
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    variable_de_ruta_de_directorio,
    image_size = (300, 300),
    batch_size = 32,
    validation_split=0.2,
    subset="training",
    seed=100)
En esta lectura y carga de imagenes se utilizan varios parametros para redimensionar las imagenes a 300x300 pixles, batch_size para definir el tamaño del lote de imagenes para el entrenamiento, validation_split que es la proporción de datos que se utilizarán como conjunto de validación y la semilla para garantizar la reproducibilidad de los resultados.

# Las imagenes ya estan clasificadas por clases y se hace la lectura y guardado de esas clases en una variable, se imprime la informacion de esas clases y se comprueba que se haya guardado de forma correcta
clases = train_data.class_names
print("Clases Entrenamiento:", clases)

# Se define el modelo CNN de redes neuronales por capas
tot_clases = len(clases)

model = keras.Sequential([
    layers.Rescaling(1./255 , input_shape = (300,300,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding='same',activation='LeakyReLU'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='LeakyReLU'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='LeakyReLU'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128,activation='LeakyReLU'),
    layers.Dense(tot_clases)
])
Se utilizan parametros para normalizar las imagenes, se utiliza capas por filtors (16, 32 y 64), se utiliza la funcion LeakyReLU que puede ayudar a evitar la saturación de la neurona, lo que puede mejorar el rendimiento de la red neuronal y se utiliza un dropout para reducir un sobre ajuste del modelo.

# Se compila al modelo de red neuronal y se aplica un optimizador para el modelo
model.compile(optimizer = 'adam',
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             metrics = ['accuracy'])
Se realiza la compilacion del modelo propuesto, se aplica el optimizador adam para reducir el error del modelo y se define tambien una funcion de perdidas para multuclase.

# Se realiza el entrenamiento del modelo con 20 epocas
entrenamiento = model.fit(
train_data,
epochs =20
Durante las 20 epocas que se utiliza para el entrenamiento del modelo se observa que en cada interaccion se aumenta cada vez la excatitud para al final del modelo obtener una exactitud superior al 90%.

# Se realiza el calculo de exactitud del modelo mediante una evaluacion entre el modelo entrenado y el conjunto de datos de prueba, en el que se obtiene 91% de exactitud
loss, accuracy = model.evaluate(test_data)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Por ultimo se procede con la grafica de las matrices de confusion, para ello se utiliza el siguiente codigo

y_true = []
y_pred = []

for images, labels in test_data:
    y_true.extend(labels.numpy())
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
cm = confusion_matrix(y_true, y_pred)
class_labels = train_data.class_names
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, cmap=plt.cm.Blues)
ax.set_xlabel("Prediccion")
ax.set_ylabel("Real")
ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center", color="white")
plt.show()

