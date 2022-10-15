import numpy as np
import time
from sklearn.svm import SVC
from mpi4py import MPI
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hacer este programa con -n 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


start_time = time.perf_counter()
# |---------------------------------|
# |  Parte secuencial del programa  |
# |---------------------------------|
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

# Al rank 0 se le va a asignar evaluar los random forest
if rank==0:
	# crear conjunto de hiperparametros, no se hace el split
	# por que se van a evaluar los hiperparametros 
	hyperparameters=[]
	for criterion in ['gini','entropy']:
		for trees in range(10, 210):
			clf = RandomForestClassifier(n_estimators=trees, criterion=criterion)
			clf.fit(X_train, y_train)
			y_pred=clf.predict(X_test)
			acc=accuracy_score(y_pred, y_test)

if rank==1:
	for kernel in ['linear', 'poly', 'rbf']:
		for gamma in ['auto', 'scale']:
			for C in range(1, 31):
				clf = SVC(kernel=kernel, gamma=gamma, C=C)
				clf.fit(X_train, y_train)
				y_pred = clf.predict(X_test)
				acc=accuracy_score(y_pred, y_test)


finish_time = time.perf_counter()
print(f"Program in rank {rank} finished in {finish_time-start_time} seconds")