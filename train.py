import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


def main(train=False):
    if train:
        df = pd.read_csv( 'diabetes_data.csv' )
        le = preprocessing.LabelEncoder()
        df_new = df.drop( columns=['Age'] )
        df_new = df_new.apply( le.fit_transform )
        df_new['Age'] = df['Age']
        X1 = df_new.drop( columns=['class'] )
        y1 = df_new['class']

        best_feature = SelectKBest( score_func=chi2, k=10 )
        fit = best_feature.fit( X1, y1 )
        dataset_scores = pd.DataFrame( fit.scores_ )
        dataset_cols = pd.DataFrame( X1.columns )
        featurescores = pd.concat( [dataset_cols, dataset_scores], axis=1 )
        featurescores.columns = ['column', 'scores']
        top_features = 5
        featurescores.nlargest( top_features, 'scores' ).to_csv( 'top_features.csv', index=False)
        df_new.to_csv( 'train_df.csv', index=False )
        X = df_new[
            featurescores.nlargest( top_features,
                                    'scores' ).column.values.tolist()]  # only using top 10 factors prediction
        y = df_new['class']
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0 )
        ss = StandardScaler()
        X_train = ss.fit_transform( X_train )
        X_test = ss.transform( X_test )
        for i in range( 1, 100 ):
            rc = RandomForestClassifier( n_estimators=i, criterion='entropy', random_state=0, n_jobs=-1 )
            rc.fit( X_train, y_train )

        accuracies = cross_val_score( estimator=rc, X=X_train, y=y_train, cv=10 )
        print( "accuracy on train set is  {:.2f} %".format( accuracies.mean() * 100 ) )
        print( "std is {:.2f} %".format( accuracies.std() * 100 ) )
        rf_predict = rc.predict( X_test )
        print( "accuracy on test set is  {:.2f} %".format( accuracy_score( rf_predict, y_test ) * 100 ) )
        print( confusion_matrix( rf_predict, y_test ) )
        print( classification_report( rf_predict, y_test ) )
        filename = 'model.sav'
        joblib.dump( rc, filename )
    else:
        pass

if __name__ == '__main__':
    main(train=True)
    # main()
