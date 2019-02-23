from __future__ import division
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)
import warnings
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import sys
import keras
import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
import ml_functions as mlf
if not sys.warnoptions:
    warnings.simplefilter("ignore")

NOTEBOOK = 0

def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    large_grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    small_grid = {
   'RF':{'n_estimators': [200], 'max_depth': [5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5], 'n_jobs':[-1]},
   'LR': { 'penalty': ['l1','l2'], 'C': [0.01,0.1,1]},
   'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
   'ET': { 'n_estimators': [200], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5], 'n_jobs':[-1]},
   'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
   'GB': {'n_estimators': [200], 'learning_rate' : [0.01,0.1],'subsample' : [0.1,0.5], 'max_depth': [5]},
   'NB' : {},
   'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10], 'max_features': [None,'sqrt','log2'],'min_samples_split': [2,5]},
   'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
   'KNN' :{'n_neighbors': [1,5,10],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
          }

#     small_grid = {
#     'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
#     'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
#     'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
#     'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
#     'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
#     'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
#     'NB' : {},
#     'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
#     'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
#     'KNN' :{'n_neighbors': [1,5,10],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
#            }

    test_grid = {
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()



def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('Model Type','Classifier', 'Parameters', 'AUC-ROC','Baseline', 'Precision at 1','Precision at 2','Precision at 5', 'Precision at 10', 'Precision at 20','Precision at 30','Precision at 50','Recall at 1','Recall at 2','Recall at 5','Recall at 10','Recall at 20','Recall at 30','Recall at 50','F1 at 1','F1 at 2','F1 at 5','F1 at 10','F1 at 20','F1 at 30','F1 at 50', 'Confusion_Matrix'))

    for n in range(1, 2):
        # create training and valdation sets
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    y_pred = clf.predict(X_test)
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,100.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 1.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 2.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 30.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 50.0),
                                                       2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 1.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 1.0)),
                                                       2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 2.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 2.0)),
                                                      2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 5.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 5.0)),
                                                      2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 10.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 10.0)),
                                                      2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 20.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 20.0)),
                                                      2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 30.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 30.0)),
                                                      2*(precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0)*recall_at_k(y_test_sorted,y_pred_probs_sorted, 50.0))/(precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0) + recall_at_k(y_test_sorted,y_pred_probs_sorted, 50.0)),
                                                      confusion_matrix(y_test, y_pred)]
                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df

from IPython.display import display

def run_simple_loop(features, label, models_to_run):

    lr_model1_aucroc_list = []
    lr_model2_aucroc_list = []
    lr_model3_aucroc_list = []
    lr_model4_aucroc_list = []
    lr_model5_aucroc_list = []
    lr_model6_aucroc_list = []

    nb_model1_aucroc_list = []

    rf_model1_aucroc_list = []
    rf_model2_aucroc_list = []
    rf_model3_aucroc_list = []
    rf_model4_aucroc_list = []
    rf_model5_aucroc_list = []
    rf_model6_aucroc_list = []
    rf_model7_aucroc_list = []
    rf_model8_aucroc_list = []

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(features):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = label.loc[train_index], label.loc[test_index]

        scaler_x = preprocessing.StandardScaler().fit(X_train)  #Regularize feature data
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)

        y_train = np.ravel(y_train,order='C')
        grid_size = 'small'
        clfs, grid = define_clfs_params(grid_size)

        # define models to run
    #     models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']
    #     models_to_run=['RF','DT', 'LR', 'NB']
#         models_to_run=['LR','NB','RF']


        # call clf_loop and store results in results_df
        results_df = clf_loop(models_to_run, clfs, grid, X_test, X_train, y_test, y_train)
        display(results_df)

        #Evaluation Metrics (averaging auc-roc for each k-fold of each model)
        lr_model1_aucroc_list.append(results_df.iloc[0][3])
        lr_model2_aucroc_list.append(results_df.iloc[1][3])
        lr_model3_aucroc_list.append(results_df.iloc[2][3])
        lr_model4_aucroc_list.append(results_df.iloc[3][3])
        lr_model5_aucroc_list.append(results_df.iloc[4][3])
        lr_model6_aucroc_list.append(results_df.iloc[5][3])
        nb_model1_aucroc_list.append(results_df.iloc[6][3])
        rf_model1_aucroc_list.append(results_df.iloc[7][3])
        rf_model2_aucroc_list.append(results_df.iloc[8][3])
        rf_model3_aucroc_list.append(results_df.iloc[9][3])
        rf_model4_aucroc_list.append(results_df.iloc[10][3])
        rf_model5_aucroc_list.append(results_df.iloc[11][3])
        rf_model6_aucroc_list.append(results_df.iloc[12][3])
        rf_model7_aucroc_list.append(results_df.iloc[13][3])
        rf_model8_aucroc_list.append(results_df.iloc[14][3])

    print("lr_model1: ", sum(lr_model1_aucroc_list)/len(lr_model1_aucroc_list))
    print("lr_model2: ", sum(lr_model2_aucroc_list)/len(lr_model2_aucroc_list))
    print("lr_model3: ", sum(lr_model3_aucroc_list)/len(lr_model3_aucroc_list))
    print("lr_model4: ", sum(lr_model4_aucroc_list)/len(lr_model4_aucroc_list))
    print("lr_model5: ", sum(lr_model5_aucroc_list)/len(lr_model5_aucroc_list))
    print("lr_model6: ", sum(lr_model6_aucroc_list)/len(lr_model6_aucroc_list))
    print("")
    print("nb_model1: ", sum(nb_model1_aucroc_list)/len(nb_model1_aucroc_list))
    print("")
    print("rf_model1: ", sum(rf_model1_aucroc_list)/len(rf_model1_aucroc_list))
    print("rf_model2: ", sum(rf_model2_aucroc_list)/len(rf_model2_aucroc_list))
    print("rf_model3: ", sum(rf_model3_aucroc_list)/len(rf_model3_aucroc_list))
    print("rf_model4: ", sum(rf_model4_aucroc_list)/len(rf_model4_aucroc_list))
    print("rf_model5: ", sum(rf_model5_aucroc_list)/len(rf_model5_aucroc_list))
    print("rf_model6: ", sum(rf_model6_aucroc_list)/len(rf_model6_aucroc_list))
    print("rf_model7: ", sum(rf_model7_aucroc_list)/len(rf_model7_aucroc_list))
    print("rf_model8: ", sum(rf_model8_aucroc_list)/len(rf_model8_aucroc_list))

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=True):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)

    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))

    print("")

    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)

def run_deep_learning_loop(features, label):
    kf = KFold(n_splits=5)
    keras_scores_nt_coding = []

    for train_index, test_index in kf.split(features):
    #         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = label.loc[train_index], label.loc[test_index]

        scaler_x = preprocessing.StandardScaler().fit(X_train)  #Regularize feature data
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
        history_nt_coding = model.fit(X_train, y_train, epochs=10, batch_size=32)
        score = model.evaluate(X_test, y_test, batch_size=128)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        keras_scores_nt_coding.append(score[1] * 100)
        print(keras_scores_nt_coding)
        plot_history(history_nt_coding)

def run_deep_learning_loop_one_hot(features, label):
    kf = KFold(n_splits=5)
    keras_scores_nt_coding = []

    for train_index, test_index in kf.split(features):
    #         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = label.loc[train_index], label.loc[test_index]

        scaler_x = preprocessing.StandardScaler().fit(X_train)  #Regularize feature data
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(19, activation='softmax'))
        model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

        # One-hot encoding
        one_hot_labels_y_train = keras.utils.to_categorical(y_train, num_classes=19)
        one_hot_labels_y_test = keras.utils.to_categorical(y_test, num_classes=19)

        history_nt_coding = model.fit(X_train, one_hot_labels_y_train, epochs=10, batch_size=32)
        score = model.evaluate(X_test, one_hot_labels_y_test, batch_size=32)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        keras_scores_nt_coding.append(score[1] * 100)
        print(keras_scores_nt_coding)
        plot_history(history_nt_coding)

def clf_loop2(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('Model Type','Accuracy','Confusion_Matrix'))

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        print(models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_probs = list(clf.predict_proba(X_test)[:,1])
                results_df.loc[len(results_df)] = [models_to_run[index] + str(p),
                                                   accuracy_score(y_test,y_pred),
                                                   confusion_matrix(y_test, y_pred)]
            except IndexError as e:
                print('Error:',e)
                continue
    return results_df

def run_simple_loop2(features, label, models_to_run):

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(features):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = label.loc[train_index], label.loc[test_index]

        scaler_x = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)

        y_train = np.ravel(y_train,order='C')
        grid_size = 'small'
        clfs, grid = define_clfs_params(grid_size)

        results_df = clf_loop2(models_to_run, clfs, grid, X_test, X_train, y_test, y_train)
        display(results_df)
