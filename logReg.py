from sklearn.linear_model import LogisticRegression

# define method for printing train and test accuracy score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from desiredTime import data_x_train, data_y_train, data_x_test, data_y_test


def print_score(clf, data_x_train, data_y_train, data_x_test, data_y_test, train=True):
    if train:
        print("train Results:\n")
        print("accuracy score:{0:4f}\n".format(accuracy_score(data_y_train, clf.predict(data_x_train))))
        print("Classification Report: \n {}\n".format(classification_report(data_y_train, clf.predict(data_x_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(data_y_train, clf.predict(data_x_train))))

