import numpy as np
import time
from mpi4py import MPI
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.perf_counter()
# |---------------------------------|
# |  Parte secuencial del programa  |
# |---------------------------------|
splits=[]
for i in range(size):
	splits.append([])

cont=0
for lr in range(1,3):
	lr=lr/100000.0
	for solver in ['adam', 'sgd', 'lbfgs']:
		splits[cont].append([lr, solver])
		cont+=1
		if cont==size:
			cont=0


X, y = make_classification(n_samples=200, n_classes=2, n_clusters_per_class=1)
#iris=load_iris()
#X=iris.data
#y=iris.target
# Se particiona el conjunto en 80-20 para la evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20)
# |---------------------------------|
# |  Parte secuencial del programa  |
# |---------------------------------|



#for hyp in splits[rank]:
#	clf = MLPClassifier(hidden_layer_sizes=(200,200,), max_iter=2000, batch_size=16, learning_rate_init=hyp[0], solver=hyp[1])
#	clf.fit(X_train, y_train)
#	y_pred = clf.predict(X_test)
#	acc=accuracy_score(y_pred, y_test)
clf = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=2000, batch_size=16, learning_rate_init=0.00001, solver='adam')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc=accuracy_score(y_pred, y_test)
print(acc)