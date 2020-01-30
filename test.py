from load_data import data
from recommender import algo
from sklearn.externals import joblib

import time

start = time.time()
trainingSet = data.build_full_trainset()
algo.fit(trainingSet)

filename = 'finalized_model.sav'
joblib.dump(algo, filename)

print('time: ', time.time() - start)

loaded_model = joblib.load(filename)
print(loaded_model.predict('B', 3).est)

# for x in range(1, 10):
#     print('score', x, ': ', loaded_model.predict(1, x).est)