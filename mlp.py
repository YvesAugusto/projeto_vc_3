from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     cross_val_predict, cross_val_score, cross_validate)
from sklearn.neural_network import MLPClassifier

from tools.data import load_dataset, show_image, split_data
from tools.extractors import (extract_components_for_dataset,
                              extract_hu_moments_from_dataset,
                              extract_laplacian_from_dataset)
import numpy as np

CM = np.zeros((10, 10))

def confusion_matrix_scorer(clf, X, y):

    global CM

    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    CM += cm

    return np.trace(cm)/np.sum(cm)

parameters = {
    'activation': "relu",
    'learning_rate': 'invscaling',
    'power_t': 0.25,
    'hidden_layer_sizes': [10, 6],
    'max_iter': 500
}

mlp = MLPClassifier(**parameters)
clf = GridSearchCV(mlp, parameters)

X, Y = load_dataset()

extractors = [
    extract_hu_moments_from_dataset, 
    extract_components_for_dataset, 
    extract_laplacian_from_dataset
]

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

for f in extractors:

    X_ = f(X)
    cv_results = cross_validate(mlp, X_, Y, cv=cv,
                            scoring=confusion_matrix_scorer)

    print(cv_results)
    print(CM/10)

    CM = np.zeros((10, 10))
