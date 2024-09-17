from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt



class Model:
    def __init__(self, model_type='svm'):
        if model_type == 'svm':
            self.model = SVC(probability=True)
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        if self.model_type == 'svm':
            grid = [
                {
                    'C': [1, 10, 100], 
                    'gamma': [0.1, 0.01, 0.001], 
                    'kernel': ['rbf', 'poly'], 
                    'degree': [2, 3, 4], 
                    'coef0': [1]
                },
            ]
            clf_gridsearch = GridSearchCV(estimator=self.model, param_grid=grid, cv=5, scoring='roc_auc')
            clf_gridsearch.fit(X, y)
            # Metrics 
            # Print best score and best parameters
            print("Best score = ", clf_gridsearch.best_score_)
            clf = clf_gridsearch.best_estimator_
            print(f"Best model: {clf}")
            # Calculate score in training and test set
            score_train = clf.score(X, y)
            print("Score in training set = ", score_train)
            score_test = clf.score(X, y)
            print("Score in test set = ", score_test)
            return clf

        if self.model_type == 'rf': # TODO random forest fit
            self.model.fit(X, y)
    
    def predict(self, X, y):
        if self.model_type == 'svm':
            # ROC
            y_test_proba_predict  = clf.predict_proba(X)
            fpr, tpr, _ = roc_curve(y, y_test_proba_predict[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, "-o", color='darkorange',
                    lw=2, label='model (AUC = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = 'random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC at test')
            plt.legend(loc="lower right")
            ## Calculamos estadístico AUC en train y test
            print(f"AUC at test: {roc_auc_score(y, clf.predict_proba(X)[:, 1])}")
            ## Calculamos también precisión y recall en test
            print(f"\ntest precision: {precision_score(y, clf.predict(X))}")
            print(f"Recall at test: {recall_score(y, clf.predict(X))}")
            print(f"Confusion matrix: \n{confusion_matrix(y, clf.predict(X))}")
        
        if self.model_type == 'rf': # TODO random forest predict
            self.model.predict(X) 
        
        