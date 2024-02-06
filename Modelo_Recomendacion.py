import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise.model_selection import cross_validate

# Cargar el conjunto de datos
df = pd.read_csv('data/df_user_items.csv')

# Preparar los datos
# Eliminar columnas innecesarias
df = df.drop(columns=['items_count', 'user_url', 'item_name'])

# Crear una matriz de usuario-ítem utilizando scipy.sparse
user_id_unique = df['user_id'][df['playtime_forever'] > 0].unique()
item_id_unique = df['item_id'][df['playtime_forever'] > 0].unique()

row_indices = np.searchsorted(user_id_unique, df['user_id'][df['playtime_forever'] > 0])
col_indices = np.searchsorted(item_id_unique, df['item_id'][df['playtime_forever'] > 0])
data = np.ones(len(row_indices))

user_item_matrix = coo_matrix((data, (row_indices - 1, col_indices - 1)), shape=(len(user_id_unique), len(item_id_unique)))
user_item_matrix = user_item_matrix.tocsr()

# Crear un objeto Reader para el conjunto de datos
reader = Reader(rating_scale=(0, df['playtime_forever'].max()))

# Crear un objeto Dataset utilizando el DataFrame de usuario-ítem y el objeto Reader
data = Dataset.load_from_folds([pd.DataFrame(user_item_matrix.toarray(), columns=df['item_id'][df['playtime_forever'] > 0]), reader], reader)

# Entrenar el modelo de recomendación
algo = KNNBasic()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Función de recomendación para un usuario dado
def recommend(user_id, algo, n=10):
    user_id = int(user_id)
    user_idx = df.index[df['user_id'] == user_id].tolist()[0]
    predictions = algo.predict(user_idx, df['item_id'][df['playtime_forever'] > 0].tolist())
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    recommended_items = [str(x.i) for x in sorted_predictions[0:n]]
    return recommended_items

# Hacer una recomendación para un usuario
recommended_items = recommend(1, algo)
print(recommended_items)