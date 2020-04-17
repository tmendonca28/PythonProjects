from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()


# X -> data
# y -> target
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

# fit the model on training data
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])

print(knn.score(X_test, y_test))

# prediction = knn.predict(X_new)
# print(prediction)