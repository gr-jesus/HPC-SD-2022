import numpy as np
import time
from mpi4py import MPI
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.perf_counter()
# |---------------------------------|
# |  Parte secuencial del programa  |
# |---------------------------------|
# crear conjunto de hiperparametros
hyperparameters=[]
for criterion in ['gini','entropy']:
    for trees in range(10, 210):
        hyperparameters.append([trees, criterion])
# Se crean las particiones para cada proceso
splits=np.split(np.array(hyperparameters), size)

# Se carga el conjunto de datos
iris=datasets.load_iris()
X=iris.data
y=iris.target
# Se particiona el conjunto en 80-20 para la evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20)
# |---------------------------------|
# |  Parte secuencial del programa  |
# |---------------------------------|

for hyp in splits[rank]:
	clf = RandomForestClassifier(n_estimators=int(hyp[0]), criterion=hyp[1])
	clf.fit(X_train, y_train)
	y_pred=clf.predict(X_test)
	#print(hyp[0],',',hyp[1],',', accuracy_score(y_pred, y_test))
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")