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


def main(train=False, classifier = None):
    if train:
        df = pd.read_csv( 'diabetes_data.csv' )
        df = df[['Polydipsia', 'Polyuria', 'sudden weight loss', 'partial paresis', 'Gender', 'Age','class']]
        X = df.drop(columns=['class'])
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        num_feat = ['Age']
        cat_feat = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'partial paresis']
        numeric_transformer = Pipeline(steps=[('Standard scalar', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        col_transformer = ColumnTransformer(transformers=[('numeric_preprocess', numeric_transformer, num_feat),
                                                          ('categorical_preprocess', categorical_transformer, cat_feat)],
                                                          remainder='drop', n_jobs=-1)
        if classifier == 'Random forest':
           pipeline_rf = Pipeline([
               ('preprocess_columns', col_transformer),
               ('random_forest_classifier', RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1))
           ])
           pipeline_rf.fit(X_train, y_train)
           print(classification_report( pipeline_rf.predict(X_test), y_test ))
           filename = 'model_random_forest.sav'
           joblib.dump( pipeline_rf, filename )
        elif classifier== 'KNN':
            pipeline_knn = Pipeline([
                ('preprocess_columns', col_transformer),
                ('KNN', KNeighborsClassifier(n_neighbors=10))
            ])
            pipeline_knn.fit(X_train, y_train)
            print(classification_report(pipeline_knn.predict(X_test), y_test))
            filename = 'model_KNN.sav'
            joblib.dump(pipeline_knn, filename)
        elif classifier == 'SVC':
            pipeline_svc = Pipeline([
                ('preprocess_columns', col_transformer),
                ('support_vectorc_classifier', SVC(gamma='auto'))
            ])
            pipeline_svc.fit(X_train, y_train)
            print(classification_report(pipeline_svc.predict(X_test), y_test))
            filename = 'model_SVC.sav'
            joblib.dump(pipeline_svc, filename)
    else:
        if classifier == 'Random forest':
            pipeline_rf = joblib.load('model_random_forest.sav')
            df = pd.read_csv('diabetes_data.csv')
            X = df.drop(columns=['class'])
            y = df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            print(classification_report(pipeline_rf.predict(X_test), y_test))
        elif classifier == 'KNN':
            pipeline_knn = joblib.load('model_KNN.sav')
            df = pd.read_csv('diabetes_data.csv')
            X = df.drop(columns=['class'])
            y = df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            print(classification_report(pipeline_knn.predict(X_test), y_test))
        elif classifier == 'SVC':
            pipeline_svc = joblib.load('model_SVC.sav')
            df = pd.read_csv('diabetes_data.csv')
            X = df.drop(columns=['class'])
            y = df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            print(classification_report(pipeline_svc.predict(X_test), y_test))
        else:
            print('Choose a classifier to load from "Random forest", "KNN" or "SVC"')


if __name__ == '__main__':
    main(train=True, classifier = 'SVC')
    main(train=True, classifier='Random forest')
    main(train=True, classifier='KNN')
    main()
