from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict
import pandas as pd
import os
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


# Asignación de valores de las variables de entorno.
db_host = os.environ.get("DB_HOST", "valor_por_defecto_host")
db_port = os.environ.get("DB_PORT", "valor_por_defecto_puerto")
db_user = os.environ.get("DB_USER", "valor_por_defecto_usuario")
db_password = os.environ.get("DB_PASSWORD", "valor_por_defecto_contraseña")
csv_file_path = os.environ.get("CSV_FILE_PATH", "valor_por_defecto_ruta_csv")

app = FastAPI()

df = pd.read_csv('data/dataframe_final.csv')


@app.get('/')
def index():
    return {'API de consulta a una base datos de Steam Games'}


@app.get("/PlayTimeGenre/{genero}")
async def playtime_genre(genero: str):

    # Capitalizar es para tener una formato consistente.
    genero = genero.capitalize()

    # Solo se incluyen las filas con el género especificado.
    df_genero = df[df[genero] == 1]
    
    # Hacemos una verificación de los datos y ver si hay disponibilidad para el género especificado.
    if df_genero.empty:
        raise HTTPException(status_code=404, detail="No se encontraron datos para el género especificado.")

    """Función que devuelve el `año` con mas horas jugadas para un género dado."""

    # Agrupa el DataFrame filtrado por 'year' y calcula la suma de 'playtime_forever' para cada año.
    year_df_playtime = df_genero.groupby('year')['playtime_total_hours'].sum().reset_index()

    # Verifica si no hay datos de tiempo de juego para el género especificado.
    if year_df_playtime.empty:
        return {"error": f"No hay datos de tiempo de juego para el género '{genero}'."}

    # Encuentra el año con el tiempo de juego total máximo.
    max_playtime_year = year_df_playtime.loc[year_df_playtime['playtime_total_hours'].idxmax(), 'year']

    # Devuelve la respuesta en el formato especificado.
    return {"Año de lanzamiento con más horas jugadas para Género": int(max_playtime_year)}


@app.get("/UserForGenre/{genero}")
async def user_for_genre(genero: str):

    # Capitalizar es para tener una formato consistente.
    genero = genero.capitalize()
    
    # Solo se incluyen las filas con el género especificado.
    df_genero = df[df[genero] == 1]
    
    if df_genero.empty:
        raise HTTPException(status_code=404, detail="No se encontraron datos para el género especificado.")

    """Función que devuelve el usuario con más horas jugadas para un género dado."""

    # Encuentra el usuario con más horas jugadas en ese género.
    usuario_max_horas = df_genero.loc[df_genero['playtime_total_hours'].idxmax(), 'user_id']
    
    # Calcula la suma del tiempo jugado por año para ese género.
    df_tiempo_juego_anio = df_genero.groupby('year')['playtime_total_hours'].sum().reset_index()

    # Renombra las columnas para que coincidan con lo esperado.
    df_tiempo_juego_anio = df_tiempo_juego_anio.rename(columns={'year': 'anio', 'playtime_total_hours': 'horas'})
    
    # Convierte el DataFrame a una lista de diccionarios con orientación 'records'.
    lista_tiempo_juego = df_tiempo_juego_anio.to_dict(orient='records')
    
    # Crear un diccionario con la información del usuario con más horas jugadas y la lista de horas jugadas por año.
    resultado = {
        "Usuario con más horas jugadas para el género " + genero: usuario_max_horas,
        "Horas jugadas": lista_tiempo_juego
    }
    
    # Devuelve el diccionario como resultado de la función.
    return resultado


@app.get("/UsersRecommend/{year}")
async def users_recommend(year: int):

    """Función que devuelve los 3 juegos más recomendados para un año dado."""

    # Filtramos el DataFrame donde la columna 'year' es igual a year, la columna 'recommend' es True y la columna 'sentiment_analysis' tiene valores 1 o 2. 

    df_filtrado = df[(df['year'] == year) & (df['recommend'] == True) & (df['review'].isin([1, 2]))]

    # Verifica si no se encontraron resultados
    if df_filtrado.empty:
        raise HTTPException(status_code=404, detail="No se encontraron juegos recomendados para el año dado.")

    # Ordena el DataFrame por la columna 'review' de manera descendente
    df_ordenado = df_filtrado.sort_values(by='review', ascending=False)

    # Toma los primeros 3 juegos del DataFrame ordenado
    top_3_resenias = df_ordenado.head(3)

    # Crea el resultado en el formato deseado
    resultado = [
        {"Puesto 1": top_3_resenias.iloc[0]['title']},
        {"Puesto 2": top_3_resenias.iloc[1]['title']},
        {"Puesto 3": top_3_resenias.iloc[2]['title']}
    ]

    if len(top_3_resenias) < 3:
        return [{"Puesto 1": top_3_resenias.iloc[0]['title']},
                {"Puesto 2": top_3_resenias.iloc[1]['title']},
                {"Puesto 3": "No hay suficientes juegos recomendados para este año."}]
    # Devuelve el resultado como una lista de diccionarios
    return resultado


@app.get("/UsersNotRecommend/{year}")
async def users_worst_developer(year: int):

    """Función que devuelve los 3 juegos menos recomendados para un año dado."""

    # Filtrado del DataFrame 
    df_filtrado = df[(df['year'] == year) & (df['recommend'] == False) & (df['review'] == 0)]
    
    if df_filtrado.empty:
        raise HTTPException(status_code=404, detail="No se encontraron juegos recomendados para el año dado.")
    
    # Calcula el puntaje para cada desarrolladora
    puntajes = []
    for desarrolladora, grupo in df_filtrado.groupby('developer'):
        total_negativos = len(grupo)
        puntajes.append({"nombre": desarrolladora, "puntaje": total_negativos})

    # Ordena desarrolladoras por puntaje
    desarrolladoras_ordenadas = sorted(puntajes, key=lambda x: x["puntaje"], reverse=True)

    # Da el top 3
    top_3 = [{"Puesto " + str(i + 1): desarrolladora["nombre"]} for i, desarrolladora in enumerate(desarrolladoras_ordenadas[:3])]

    if len(desarrolladoras_ordenadas) < 3:
        return [{"Puesto 1": "No hay suficientes desarrolladores con juegos no recomendados para este año."},
                {"Puesto 2": ""},
                {"Puesto 3": ""}]

    return top_3


@app.get("/sentiment_analysis/{empresa_desarrolladora}")
async def sentiment_analysis(empresa_desarrolladora: str):

    """Función que devuelve la cantidad de comentarios positivos, negativos y neutrales para un año dado."""

    # Filtra el DataFrame por empresa desarrolladora
    df_empresa = df[df['developer'] == empresa_desarrolladora]

    # Cuenta la cantidad de reseñas por cada categoría de sentimiento
    counts = df_empresa['review'].value_counts()
    negativas = counts.get(0, 0)
    neutrales = counts.get(1, 0)
    positivas = counts.get(2, 0)

    # Crea el diccionario de retorno con el nombre de la desarrolladora y los resultados del análisis de sentimiento
    resultado = {empresa_desarrolladora: [f"Negative = {negativas}", f"Neutral = {neutrales}", f"Positive = {positivas}"]}

    return resultado


# Columnas relevantes
columns_use = ['user_id', 'item_id', 'review', 'title']
data = df[columns_use].fillna('')  # Llena valores NaN con cadenas vacías para evitar problemas con el modelo

# Verifica que la columna 'review' contenga números y no valores NaN
data['review'] = data['review'].fillna(1)  # Por ejemplo, llenar NaN con 1 para considerarlos neutrales

# Crea y entrena el modelo Nearest Neighbors
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model.fit(data[['review']])  # Utiliza directamente la columna 'review' como características

@app.get("/recomendacion_juego/{item_id}", response_model=List[dict])
async def recomendacion_juego(user_review:int):
    try:
        # Encuentra vecinos más cercanos basados en la columna 'review'
        _, indices = model.kneighbors([[user_review]])

        # Obtiene las recomendaciones
        recommendations = data.loc[indices[0], ['title', 'review']]

        return recommendations.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
