import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer

def ordinalencoder_transformer(X_train, X_test, features):
    ordinalEncoder = OrdinalEncoder()
    for feature in features:
        ordinalEncoder.fit(X_train[[feature]])
        for s_ in [X_train, X_test]:
            s_[feature] = ordinalEncoder.transform(s_[[feature]])
    return X_train, X_test


def one_hot_encode(df, object_cols):
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), object_cols),
        remainder='passthrough')

    df_encoded_array = preprocessor.fit_transform(df)
    feature_names_encoded = preprocessor.named_transformers_['onehotencoder'].get_feature_names_out(object_cols)
    feature_names_passthrough = [col for col in df.columns if col not in object_cols]
    all_feature_names = list(feature_names_encoded) + feature_names_passthrough
    df_final = pd.DataFrame(df_encoded_array, columns=all_feature_names, index=df.index)

    return df_final
