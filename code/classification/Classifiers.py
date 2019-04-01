import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.metrics import classification_report

# classifiers = [
#     {
#         'name': 'RandomForest',
#         'clf': RandomForestClassifier(),
#         'grid': {
#             'n_estimators': [100, 1000],
#             'max_depth': [2, 3]
#         }
#     },
# ]

metrics = ['accuracy', 'precision', 'f1', 'roc_auc', 'recall']


class Classifiers:
    def __init__(self, feature_df: pd.DataFrame, labels, classifiers):
        self.df = feature_df
        # Extract the dataframe as a numpy array
        self.data = feature_df.to_numpy()
        self.labels = labels

        self.classifiers = classifiers

    def information_gain(self):
        info = mutual_info_classif(self.data, self.labels)

        print("Information gain of whole dataset")
        print(info)

        return info

    def chi2_stats(self):
        pvals = chi2(self.data, self.labels)

        print("Chi2 stats of whole dataset")
        print(pvals)

        return pvals

    def _get_clf_attributes(self, val):
        try:
            # Deconstruct classifiers settings
            clf = val['clf']
            name = val['name']

            return clf, name
        except Exception as e:
            raise e

    def _get_optimized_clf(self, val):
        # Make sure that the classifier is defined
        try:
            classifier, name = self._get_clf_attributes(val)
        except ValueError as e:
            print(e)
            return None

        # Fail fast if model or param not available
        if 'optimized_model' not in val and 'optimized_param' not in val:
            print("Classifier {} not optimized, first call .optimize() or define optimized_param.".format(name))
            return None

        # Check if optimized classifier is available
        if 'optimized_model' in val:
            clf = val['optimized_model']

        if 'optimized_param' in val:
            params = val['optimized_param']
            clf = classifier.set_params(**params)

        return clf

    def optimize(self):
        print("-- Start optimizing by grid search --")
        for _, val in enumerate(self.classifiers):
            # Make sure that the classifier is defined
            try:
                clf, name = self._get_clf_attributes(val)

                # We also need the optimization grid
                grid = val['grid']
            except Exception as e:
                print(e)
                continue

            print("Optimizing: {}".format(name))

            # Do a grid search with 10 folds
            optimize_cv = KFold(n_splits=10, shuffle=True)
            # TODO: to what metric do we refit?
            clf = GridSearchCV(estimator=clf, param_grid=grid, cv=optimize_cv, scoring=metrics, refit='roc_auc')
            clf.fit(self.data, self.labels)
            params = clf.best_params_

            print("Optimal settings {}:".format(name))
            print(params)

            # Save the results
            val['optimized_param'] = params
            val['optimized_model'] = clf.best_estimator_

        print("-- Finished optimizing -- ")

    def cross_val(self):
        print("-- Cross validation with 10-folds --")
        for _, val in enumerate(self.classifiers):
            # Get optimized classifier
            clf = self._get_optimized_clf(val)

            # Skip this one if it was not available
            if clf is None:
                continue

            # Now check performance of the classifier with cross validation
            test_cv = KFold(n_splits=10, shuffle=True)
            performance = cross_validate(estimator=clf, X=self.data, y=self.labels, scoring=metrics, cv=test_cv)

            # Average each metric over the folds
            for k, scores in performance.items():
                performance[k] = scores.mean()

            print("Cross validation performance:")
            print(performance)

        print("-- Finished cross validation --")

    def test(self):
        print("-- Performance on split: 80% train - 20% split --")
        for _, val in enumerate(self.classifiers):
            # Get optimized classifier
            clf = self._get_optimized_clf(val)

            # Split the data 80/20 in trn/tst
            trn, tst, trn_label, tst_label = train_test_split(self.data, self.labels, test_size=0.2)

            # Train classifier
            clf.fit(trn, trn_label)

            # Skip this one if it was not available
            if clf is None:
                continue

            # Make predictions with the model
            y_preds = clf.predict(tst)

            # Óutput classification report
            # TODO: is clickbait 0 or 1?
            class_names = ['class 0', 'class 1']
            report = classification_report(y_true=tst_label, y_pred=y_preds, target_names=class_names)
            print(report)

        print("-- Finished test reports --")
