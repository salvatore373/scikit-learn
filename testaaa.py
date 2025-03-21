import numpy as np
import pandas as pd

from sklearn import metrics, tree
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    # df = pd.read_csv('/Volumes/SALVATORE R/Università/PaperNLP/data_2/AAATESTTEST/test_data/band_train.csv',
    #                  index_col=0)
    # X_train = df.drop('label', axis=1)
    # y_train = df['label']
    X_train = pd.DataFrame({'a': [1] * 3 + [0] * 3 + [0] * 3 + [0] * 9,
                            'b': [0] * 3 + [1] * 3 + [0] * 3 + [0] * 9,
                            'c': [0] * 3 + [0] * 3 + [1] * 3 + [0] * 9,
                            'd': [0] * 9 + [1] * 3 + [0] * 3 + [0] * 3,
                            'e': [0] * 9 + [0] * 3 + [1] * 3 + [0] * 3,
                            'f': [0] * 9 + [0] * 3 + [0] * 3 + [1] * 3,
                            })
    y_train = np.array(['salvo'] * 9 + ['miche'] * 9)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="gini", max_depth=3)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_train)

    tree.export_graphviz(clf, '/Volumes/SALVATORE R/Università/PaperNLP/data_2/AAATESTTEST/my_tree.dot',
                         feature_names=X_train.columns, class_names=np.unique(y_train), filled=True,
                         rotate=True, rounded=True)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_train, y_pred))
