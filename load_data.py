import pandas as pd
from surprise import Dataset
from surprise import Reader

from random import randint

item_list = []
user_list = []
rating_list = []

for x in range(0, 100):
    item_list.append(x%10 + 1)
    user_list.append(x//10 + 1)
    rating_list.append(randint(1, 10))

# ratings_dict = {
#     "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
#     "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
#     "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
# }

ratings_dict = {
    "item": [1, 2, 3, 1, 2],
    "user": ['A', 'A', 'A', 'B', 'B'],
    "rating": [3, 4, 1, 3, 4],
}

# ratings_dict = {
#     "item": item_list,
#     "user": user_list,
#     "rating": rating_list,
# }

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 10))

# Loads Pandas dataframe
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
# Loads the builtin Movielens-100k data
movielens = Dataset.load_builtin('ml-100k')