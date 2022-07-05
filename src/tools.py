import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector

from operator import itemgetter

def getFeaturesEndsCertainNumber(features:list, periodNumber:int):
    selectedFeatures = []
    for feature in features:
        if feature.endswith("_6"):
            selectedFeatures.append(feature)
    
    return selectedFeatures

def getDropFeaturesGroup(duplicated_features):
    set_duplicated_features = set()
    for feature in duplicated_features:
        lstFeature = feature.split('_')[:-1]
        feature = '_'.join(lstFeature)
        set_duplicated_features.add(feature)
    print(f"set_duplicated_features: {set_duplicated_features}")
    print(f"len(set_duplicated_features): {len(set_duplicated_features)}")
    return set_duplicated_features

def addPeriodSuffix(set_duplicated_features, PERIOD_START=6, PERIOD_LENGTH=15):
    dropFeatures = []
    for featrueNameWithoutPeriod in set_duplicated_features:
        for i in range(PERIOD_START, PERIOD_START+PERIOD_LENGTH):    
            featrueNameWithPeriod = featrueNameWithoutPeriod + '_' + str(i)
            dropFeatures.append(featrueNameWithPeriod)

    print(f"dropFeatures: {dropFeatures}")
    print(f"len(dropFeatures): {len(dropFeatures)}")
    return dropFeatures

def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((x.shape[0], img_width, img_height, 1))
    for i in range(x.shape[0]):
        # print(type(x), type(x_temp), x.shape)
        x_temp[i] = np.reshape(x.iloc[i].values, (img_width, img_height, 1))

    return x_temp

def normalize_data(x_train_df, x_cv_df, x_test_df):
    mm_scaler = MinMaxScaler(feature_range=(0, 1)) # or StandardScaler?
    x_train_scaled = pd.DataFrame(mm_scaler.fit_transform(x_train_df), columns=x_train_df.columns, index=x_train_df.index)
    x_cv_scaled = pd.DataFrame(mm_scaler.transform(x_cv_df), columns=x_cv_df.columns, index=x_cv_df.index)
    x_test_scaled = pd.DataFrame(mm_scaler.transform(x_test_df), columns=x_test_df.columns, index=x_test_df.index)
    return x_train_scaled, x_cv_scaled, x_test_scaled 

def print_feature_scores(columns, f_score, p_values):
    print ("Features     ", "F-Score    ", "P-Values")
    print ("-----------  ---------    ---------")

    for i in range(0, len(columns)):
        f1 = "%4.2f" % f_score[i]
        p1 = "%2.6f" % p_values[i]
        print(columns[i].ljust(12), f1.rjust(8),"    ", p1.rjust(8))

def remove_duplicated_features(x_train, x_cv, x_test):
    x_train_T = x_train.T
    
    # find the total number of duplicated features in dataset using the sum() method
    print(f"x_train_T.duplicated().sum(): {x_train_T.duplicated().sum()}")
    x_train_with_unique_features = x_train_T.drop_duplicates(keep='first').T

    # To see the names of the duplicated columns
    duplicated_features = [dup_col for dup_col in x_train.columns if dup_col not in x_train_with_unique_features.columns]
    print(f"duplicated features: {duplicated_features}")
    # unique_features = x_train.columns
    # x_train = x_train[unique_features]
    # x_cv = x_cv[unique_features]
    # x_test = x_test[unique_features]
    print(f"=============== shape before dropping {x_train.shape} =================")
    x_train_drop = x_train.drop(duplicated_features, axis=1)
    x_cv_drop = x_cv.drop(duplicated_features, axis=1)
    x_test_drop = x_test.drop(duplicated_features, axis=1)
    print(f"=============== shape after dropping {x_train_drop.shape} =================")

    return {
        "x_train": x_train_drop,
        "x_cv": x_cv_drop, 
        "x_test": x_test_drop,
        "duplicated_features": duplicated_features
    }

def remove_Quasi_constant_features(x_train, x_cv, x_test, threshold=0.9):
    '''
    If we pass 0.01, which means that if the variance of the values in a column is less than 0.01, 
    remove that column. In other words, remove feature column where approximately 99% of the values are similar.
    '''
    constant_filter = VarianceThreshold(threshold=threshold)
    constant_filter.fit_transform(x_train)

    # to get all the features that are not constant
    colsIdx = constant_filter.get_support(indices=True)
    non_constant_columns = x_train.columns[colsIdx]
    print(f"features that are not Quasi constant: {len(non_constant_columns)}")
    constant_columns = [column for column in x_train.columns
                        if column not in non_constant_columns]
    print(f"The number of features that are Quasi constant: {len(constant_columns)}")
    print(f"features that are Quasi constant: {constant_columns}")

    # Another way to drop columns
    # x_train.drop(labels=constant_columns, axis=1, inplace=True)

    selectedCols = non_constant_columns.to_list()
    x_train_drop = x_train[selectedCols]
    x_cv_drop = x_cv[selectedCols]
    x_test_drop = x_test[selectedCols]

    return {
        "x_train": x_train_drop,
        "x_cv": x_cv_drop, 
        "x_test": x_test_drop,
        "drop_features": constant_columns
    }

def remove_constant_features(x_train, x_cv, x_test):
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit_transform(x_train)

    # to get all the features that are not constant
    colsIdx = constant_filter.get_support(indices=True)
    non_constant_columns = x_train.columns[colsIdx]
    print(f"features that are not constant: {len(non_constant_columns)}")
    constant_columns = [column for column in x_train.columns
                        if column not in non_constant_columns]
    print(f"The number of features that are constant: {len(constant_columns)}")
    print(f"features that are constant: {constant_columns}")

    # Another way to drop columns
    # x_train.drop(labels=constant_columns, axis=1, inplace=True)

    selectedCols = non_constant_columns.to_list()
    x_train_drop = x_train[selectedCols]    
    x_cv_drop = x_cv[selectedCols]
    x_test_drop = x_test[selectedCols]

    return {
        "x_train": x_train_drop,
        "x_cv": x_cv_drop, 
        "x_test": x_test_drop,
        "drop_features": constant_columns
    }

def remove_high_corr_features(x_train, x_cv, x_test, threshold=0.8, showHeatmap = False):
    correlated_features = set()
    correlation_matrix = x_train.corr()
 
    if showHeatmap:
        plt.figure(figsize = (15,10))
        sns.heatmap(correlation_matrix, annot=True)
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
                print(f"Feature '{colname}' is correlated with '{correlation_matrix.columns[j]}'")
    
    print(f"len(correlated_features): {len(correlated_features)}")
    print(f"correlated_features: {correlated_features}")
    x_train_drop = x_train.drop(labels=correlated_features, axis=1)
    x_cv_drop = x_cv.drop(labels=correlated_features, axis=1)
    x_test_drop = x_test.drop(labels=correlated_features, axis=1)

    return {
        "x_train": x_train_drop,
        "x_cv": x_cv_drop, 
        "x_test": x_test_drop,
        "drop_features": correlated_features
    }

def feature_selection(selection_method, x_train, x_cv, x_test, y_train, number_features):
    '''
    F_regression is used for regression while chi2 is used for classification.
    f_regression p_value wil calculate the linear dependancy between each regressor and the target.
    chi2 test measures dependence between stochastic variables.
    '''
    number_features = int(number_features)
    print(f"number_features: {number_features}")
    
    # list_features = list(x_train.loc[:, 'open_':].columns)
    list_features = list(x_train.columns)
    
    if selection_method == 'chi2' or selection_method == 'all':
        pass

    if selection_method == 'anova' or selection_method == 'all':
        print("******************* anova *********************")
        select_k_best = SelectKBest(f_classif, k=number_features)
        
        x_train_k = select_k_best.fit_transform(x_train, y_train)
        x_cv_k = select_k_best.transform(x_cv)
        x_test_k = select_k_best.transform(x_test)
        
        # Get f_score and p_values for the selected features
        f_score = select_k_best.scores_
        p_values = select_k_best.pvalues_
        print_feature_scores(x_train.columns, f_score, p_values)

        colsIdx = select_k_best.get_support(indices=True)
        print(f"colsIdx: {colsIdx}")
        selectedCols = x_train.columns[colsIdx].to_list()
        print(f"selected_features_anova: {selectedCols}")
        
        selected_features_anova = itemgetter(*colsIdx)(list_features)
        # print(f"selected_features_anova: {selected_features_anova}")
        print("****************************************")
    
    # information gain
    if selection_method == 'mutual_info' or selection_method == 'all':
        print("******************* mutual_info *********************")
        # feature_scores = mutual_info_classif(x_train, y_train, random_state=0)
        feature_scores = mutual_info_classif(x_train, y_train)
        feature_scores = pd.Series(feature_scores)
        feature_scores.index = x_train.columns
        feature_scores = feature_scores.sort_values(ascending=False)
        print(f"mutual_info_classif result: {feature_scores}")
        #let's plot the ordered mutual_info values per feature
        # feature_scores.sort_values(ascending=False).plot.bar(figsize=(20, 8))

        select_k_best = SelectKBest(mutual_info_classif, k=number_features)
        print(f"mutual_info_classif select_k_best result: {select_k_best}")
        x_train_k = select_k_best.fit_transform(x_train, y_train)
        x_cv_k = select_k_best.transform(x_cv)
        x_test_k = select_k_best.transform(x_test)
        print(f"mutual_info_classif x_train_k: {x_train_k}")

        # Get f_score and p_values for the selected features
        f_score = select_k_best.scores_
        print(f"mutual_info_classif f_score: {f_score}")
        # p_values = select_k_best.pvalues_     # information gain沒有p_values
        # print_feature_scores(x_train.columns, f_score, p_values)

        colsIdx = select_k_best.get_support(indices=True)
        selectedCols = x_train.columns[colsIdx].to_list()
        print(f"selected_features_mutual_info: {selectedCols}")

        selected_features_mic = itemgetter(*colsIdx)(list_features)
        # print(len(selected_features_mic), selected_features_mic)
        print("****************************************")

    if selection_method == 'stepForward':
        sfs = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1), 
           k_features=number_features, 
           forward=True, 
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5).fit(x_train, y_train)
        
        colsIdx = list(sfs.k_feature_idx_)
        print(f"sfs.k_feature_idx_: {sfs.k_feature_idx_}")      # (0, 8, 16, 20, 26, 32, 36, 40, 45, 52, 61, 62, 69, 70, 84)
        print(f"sfs.k_feature_names_: {sfs.k_feature_names_}")
        print(f"sfs.k_score_: {sfs.k_score_}")
        print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
        
        x_train_k = sfs.transform(x_train)
        x_cv_k = sfs.transform(x_cv)
        x_test_k = sfs.transform(x_test)

        selectedCols = x_train.columns[colsIdx].to_list()
        print(f"selected_features: {selectedCols}")

    if selection_method == 'all':
        print("******************* all *********************")
        common_features = list(set(selected_features_anova).intersection(selected_features_mic))
        print("common selected featues", len(common_features), common_features)
        if len(common_features) < number_features:
            raise Exception('number of common features found {} < {} required features. Increase "number_features variable"'.format(len(common_features), number_features))
        feature_idx = []
        for c in common_features:
            feature_idx.append(list_features.index(c))
        feature_idx = sorted(feature_idx[0:number_features])
        print(f"feature_idx: {feature_idx}")


    if selection_method == 'all':
        x_train = x_train[:, feature_idx]
        x_cv = x_cv[:, feature_idx]
        x_test = x_test[:, feature_idx]
    else:
        x_train = x_train[selectedCols]
        x_cv = x_cv[selectedCols]
        x_test = x_test[selectedCols]

    # print(f"Shape of x, y train {x_train.shape}, {y_train.shape}")
    # print(f"Shape of x test {x_test.shape}")
    # print(f"Shape of x validation {x_cv.shape}")
    return x_train, x_cv, x_test

def check_inf_exist(df):
    inf_count = np.isinf(df).values.sum()
    print(f"inf_count: {inf_count}")

    # printing column name where infinity is present
    print("printing column name where infinity is present")
    cols_name = list(df.columns.to_series()[np.isinf(df).any()])
    print(f"cols_name: {cols_name}")
    print(f"len(cols_name): {len(cols_name)}")

    # counting infinity in a particular column name
    for col_name in cols_name:
        c = np.isinf(df[col_name]).values.sum()
        print(f"{col_name} contains " + str(c) + " infinite values")

if __name__ == "__main__":

    # from sklearn.model_selection import train_test_split
    # f = pd.read_csv('Students2.csv')
    # X = f.iloc[:, :-1]
    # Y = f.iloc[:,  -1]
    # print(f"type(X): {type(X)}")
    # print(f"type(Y): {type(Y)}")
    # print(f"X.head(): {X.head()}")
    # print(f"Y.head(): {Y.head()}")
    
    # print('Total number of datas: ', len(X))
    # print('Total number of features: ', len(X.columns))

    # print("Ready to split training and testing set")
    # train_split = 0.8
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_split, test_size=1-train_split, 
    #                                                     random_state=2, shuffle=True)

    # # print(f"x_train.head(): {x_train.head()}")
    # # print(f"y_train.head(): {y_train.head()}")

    # print("Ready to split training and validation set")
    # train_split = 0.8
    # x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=train_split, test_size=1-train_split, 
    #                                                 random_state=2, shuffle=True)

    # # print("Before normalize data:")
    # # print(f"x_train.head(): {x_train.head()}")
    # # print(f"x_cv.head(): {x_cv.head()}")
    # # print(f"x_test.head(): {x_test.head()}")
    # x_train_scaled, x_cv_scaled, x_test_scaled = normalize_data(x_train, x_cv, x_test)
    # # print("After normalize data:")
    # # print(f"x_train_scaled.head(): {x_train_scaled.head()}")
    # # print(f"x_cv_scaled.head(): {x_cv_scaled.head()}")
    # # print(f"x_test_scaled.head(): {x_test_scaled.head()}")
    
    # print("Before feature selection:")
    # print(f"Shape of x_train {x_train.shape}")
    # print(f"Shape of x_validation {x_cv.shape}")
    # print(f"Shape of x_test {x_test.shape}")
    # x_train, x_cv, x_test = feature_selection("mutual_info", x_train, x_cv, x_test, y_train, len(x_train.columns)*0.8)
    # print("After feature selection:")
    # print(f"Shape of x_train {x_train.shape}")
    # print(f"Shape of x_validation {x_cv.shape}")
    # print(f"Shape of x_test {x_test.shape}")

    # ===============================================================================================

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    data = load_boston()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["MyNewCol_1"] = 100
    df["MyNewCol_2"] = 100
    # df["MyNewCol_10"] = 100
    # df["MyNewCol_20"] = 200

    # remove_high_corr_features(df)
    remove_duplicated_features(df, df, df)
    # remove_constant_features(df)
    # remove_Quasi_constant_features(df)