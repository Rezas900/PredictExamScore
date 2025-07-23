import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from pandas.plotting import scatter_matrix


def feature_importance_by_random_forest(dataset, N_estimator=100):
    dataset = dataset.copy()

    mode = dataset["parental_education_level"].mode()[0]
    dataset["parental_education_level"].fillna(mode, inplace=True)

    ordinalEncoder = OrdinalEncoder()
    X = ordinalEncoder.fit_transform(dataset)
    df_num = pd.DataFrame(X, columns=dataset.columns,
                          index=dataset.index)

    train_ = df_num.drop(["exam_score"], axis=1)
    test_ = df_num[["exam_score"]]

    rand_reg = RandomForestRegressor(n_estimators=N_estimator, n_jobs=-1)
    rand_reg.fit(train_, test_)
    for name, score in zip(train_.columns, rand_reg.feature_importances_):
        print(f"{name} ::::>  {score:.4f}")


def feature_importance_by_corrMatrix(dataset, targetColumn):
    dataset = dataset.copy()
    encoder = OrdinalEncoder()
    df_num = encoder.fit_transform(dataset)

    df_num = pd.DataFrame(df_num, columns=dataset.columns, index=dataset.index)
    corrMatrix = df_num.corr()
    return corrMatrix[targetColumn].sort_values(ascending=False)