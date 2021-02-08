import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

svc = 'SVC'
rf = 'Random forest classifier'
KNN = 'KNN classifier'
adaboost = 'Adaboost classifier'
gboost = 'Gradient boosting classifier'
extra_tree = 'Extra tree Classifier'

def load_evaluate_model(model):
    """
    Load and evaluate provided model on test data
    :param model: sklearn pipeline type
    :return: print classification report
    """
    df = pd.read_csv('diabetes_data.csv')
    df = df[['Polydipsia', 'Polyuria', 'sudden weight loss', 'partial paresis', 'Gender', 'Age', 'class']]
    X = df.drop(columns=['class'])
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(classification_report(model.predict(X_test), y_test))


def main(train=False, classifier=None):
    if train:
        df = pd.read_csv('diabetes_data.csv')
        df = df[['Polydipsia', 'Polyuria', 'sudden weight loss', 'partial paresis', 'Gender', 'Age', 'class']]
        X = df.drop(columns=['class'])
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        num_feat = ['Age']
        cat_feat = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'partial paresis']
        numeric_transformer = Pipeline(steps=[('Standard scalar', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        col_transformer = ColumnTransformer(transformers=[('numeric_preprocess', numeric_transformer, num_feat),
                                                          (
                                                          'categorical_preprocess', categorical_transformer, cat_feat)],
                                            remainder='drop', n_jobs=-1)
        if classifier == rf:
            pipeline_rf = Pipeline([
                ('preprocess_columns', col_transformer),
                ('random_forest_classifier', RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1))
            ])
            pipeline_rf.fit(X_train, y_train)
            print(classification_report(pipeline_rf.predict(X_test), y_test))
            filename = 'model_random_forest.sav'
            joblib.dump(pipeline_rf, filename)
        elif classifier == KNN:
            pipeline_knn = Pipeline([
                ('preprocess_columns', col_transformer),
                ('KNN', KNeighborsClassifier(n_neighbors=10))
            ])
            pipeline_knn.fit(X_train, y_train)
            print(classification_report(pipeline_knn.predict(X_test), y_test))
            filename = 'model_KNN.sav'
            joblib.dump(pipeline_knn, filename)
        elif classifier == svc:
            pipeline_svc = Pipeline([
                ('preprocess_columns', col_transformer),
                ('support_vectorc_classifier', SVC(gamma='auto'))
            ])
            pipeline_svc.fit(X_train, y_train)
            print(classification_report(pipeline_svc.predict(X_test), y_test))
            filename = 'model_SVC.sav'
            joblib.dump(pipeline_svc, filename)
        elif classifier == adaboost:
            pipeline_ada = Pipeline([
                ('preprocess_columns', col_transformer),
                ('Adaboost_stump', AdaBoostClassifier(n_estimators=50, random_state=0))
            ])
            pipeline_ada.fit(X_train, y_train)
            print(classification_report(pipeline_ada.predict(X_test), y_test))
            filename = 'model_adaboost.sav'
            joblib.dump(pipeline_ada, filename)

        elif classifier == gboost:
            pipeline_gbc = Pipeline([
                ('preprocess_columns', col_transformer),
                ('Gradient_boost_classifier',
                 GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=0))
            ])
            pipeline_gbc.fit(X_train, y_train)
            print(classification_report(pipeline_gbc.predict(X_test), y_test))
            filename = 'model_gradient_boosting.sav'
            joblib.dump(pipeline_gbc, filename)
        
        elif classifier == extra_tree:
            pipeline_etc = Pipeline([
                ('preprocess_columns', col_transformer),
                ('Extra_tree_classifier',
                 ExtraTreesClassifier(n_estimators=100, random_state=0,max_depth=3,bootstrap=True,n_jobs=-1))
            ])
            pipeline_etc.fit(X_train, y_train)
            print(classification_report(pipeline_etc.predict(X_test), y_test))
            filename = 'model_extra_tree.sav'
            joblib.dump(pipeline_etc, filename)
    else:
        if classifier == rf:
            pipeline_rf = joblib.load('model_random_forest.sav')
            load_evaluate_model(pipeline_rf)
        elif classifier == KNN:
            pipeline_knn = joblib.load('model_KNN.sav')
            load_evaluate_model(pipeline_knn)
        elif classifier == svc:
            pipeline_svc = joblib.load('model_SVC.sav')
            load_evaluate_model(pipeline_svc)
        elif classifier == adaboost:
            pipeline_ada = joblib.load('model_adaboost.sav')
            load_evaluate_model(pipeline_ada)
        elif classifier == gboost:
            pipeline_gbc = joblib.load('model_gradient_boosting.sav')
            load_evaluate_model(pipeline_gbc)
        elif classifier == extra_tree:
            pipeline_etc = joblib.load('model_extra_tree.sav')
            load_evaluate_model(pipeline_etc)

        else:
            print(f'Choose a classifier to load from {svc, rf, KNN,adaboost,gboost,extra_tree }')


if __name__ == '__main__':
    # main(train=True, classifier=svc)
    # main(train=True, classifier=rf)
    # main(train=True, classifier=KNN)
    # main(train=True, classifier=adaboost)
    # main(train=True, classifier=gboost)
    # main(train=True, classifier=extra_tree)
    main()
