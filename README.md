# Machine-Learning-Operations-MLOps-


# Machine-Learning-Operations-MLOps-


## Introducción

El Proyecto desarrollado y presentado a continuacion es parte de una base de datos de una empresa de videojuegos. En el cual se decidio que funcionalidades son esenciales en el sistema, empezando por la limpieza de la base de datos y su adaptacion, el backend (APIs, lógica del negocio), todo esto con la finalidad de dar recomendaciones personalizadas para la empresa y que se mas facil la ubicacion de informacion, poder ver los rankings de los mejores videojuegos para la empresa con la finalidad de que se pueda analizar la conveniencia de donde se importan los gastos de desarrollo de videojuegos. La empresa nos proporciono archivos JSON anidados, los cuales para un proceso mas facil los converti en CSV, los cuales podremos encontrar en mi GITHUB (https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-) en la carpeta de data(https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-/tree/master/data)


## Proceso 

1. **ETL (Extraccion, Transformacion, Carga): **

En este punto se hizo una extraccion de informacion de un archivo JSON y se convirtio en CSV, con la finalidad de tener una practicidad de manejo de los mismos. Los datos se desanidaron y se hizo una limpieza y eliminacion de los datos que no serian requeridos para el proyecto. Se fiktraron los datos nulos y los datos duplicados que furon elimnados para no tener archivos duplicados y liberar espacio de almaceniamiento de los archivos. Al terminar se obtuvo los archivos limpios guardados en (https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-/tree/master/data). Se puede acceder al ETL en (https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-/blob/master/ETL.ipynb)

2. **Feature Engineering:**

Se Creo un "Sentiment_analysis" ya con los dato limpios y cargados. En el dataset *user_reviews* se incluyen reseñas de juegos hechos por distintos usuarios. Debes crear la columna ***'sentiment_analysis'*** aplicando análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna debe reemplazar la de user_reviews.review para facilitar el trabajo de los modelos de machine learning y el análisis de datos. De no ser posible este análisis por estar ausente la reseña escrita, debe tomar el valor de `1`  (https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-/blob/master/ETL.ipynb)


3. **Desarrollo API**:

Se creo un archvio maain.py (https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-/blob/master/main.py). En el cual se creo una FastAPI con la ayuda de Python y sus bibliotecas, los punto solicitados fueron: 

<sub> Debes crear las siguientes funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).<sub/>


+ def **PlayTimeGenre( *`genero` : str* )**:
    Debe devolver `año` con mas horas jugadas para dicho género.
  
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

+ def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
                 "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

+ def **UsersRecommend( *`año` : int* )**:
   Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **UsersWorstDeveloper( *`año` : int* )**:
   Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **sentiment_analysis( *`empresa desarrolladora` : str* )**:
    Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor. 

Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}

-- Desarrollo en RENDER (https://dashboard.render.com/)  
-- FastAPI (https://fastapi-mlops.onrender.com) 

4. **EDA:**

En el EDA se hace un Analisis Exploratorio de Datos (EDA), en el cual se explora y se analisa el conjnto de datos. En el cual damos a conocer hayasgos de los datos de una mera grafica y mas visual al ojo publico, ya que no para muchos el ver codigo no es sficiente para poder comprender la informacion o lo que se hace con ella (https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-/blob/master/EDA.ipynb)

5. **Sistema de recomendación:**
Se creo un modelo de recomendacion User - Items, en el cual ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario. Con la finalidad de saber que videojuegos prefiere ese usuario y empezar a categorizar y saber cuales son los preferidos por el publico (https://github.com/MFLopezBello/Machine-Learning-Operations-MLOps-/blob/master/Modelo_recomendacion.ipynb)

6. **Video:**
[Video explicativo del proyecto](https://drive.google.com/drive/folders/1CTMf3JWZS67LPs1OFaN-1wXL-Ak0jT0i?usp=sharing).
