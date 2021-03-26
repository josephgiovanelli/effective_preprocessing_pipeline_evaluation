import numpy as np
import openml
import os
#import h2o
import pandas as pd
#from pymfe.mfe import MFE

from enum import Enum
class UndefinedOrders(Enum):
    features_rebalance = 1
    discretize_rebalance = 2

class DefinedOrders(Enum):
    first_second = 1
    second_first = 2

def extract_metafeatures(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute)

    dict = {}
    mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])

    mfe.fit(X, y)
    ft = mfe.extract()

    for i in range(0, len(ft[0])):
        dict[ft[0][i]] = ["na"] if np.isnan(ft[1][i]) else [ft[1][i]]

    return dict

def load_metafeatures(id):
    df = pd.read_csv('results_processors/meta_features/extracted-meta-features.csv')
    df = df.loc[df['id'] == id]
    df = df.drop(['id'], axis=1)
    df = df.fillna("na")
    return df

def check_existence(id, algorithm):
    acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
    features_rebalance = os.path.exists("models/features_rebalance/three_classes_id/xgboost_" + acronym + "/" + str(id))
    discretize_rebalance = os.path.exists("models/discretize_rebalance/three_classes_id/xgboost_" + acronym + "/" + str(id))
    return features_rebalance and discretize_rebalance

def predict_order(id, meta_features, algorithm, undefined_order):
    acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
    if undefined_order == UndefinedOrders.features_rebalance:
        saved_model = h2o.load_model("models/features_rebalance/three_classes_id/xgboost_" + acronym + "/" + str(id) + "/model")
    else:
        saved_model = h2o.load_model("models/discretize_rebalance/three_classes_id/xgboost_" + acronym + "/" + str(id) + "/model")

    prediction = saved_model.predict(h2o.H2OFrame(meta_features, na_strings=["na"]))["predict"]
    if prediction == "no_order":
        if undefined_order == UndefinedOrders.features_rebalance:
            if acronym == "rf":
                return DefinedOrders.first_second
            elif acronym == "knn":
                return DefinedOrders.second_first
            elif acronym == "nb":
                return DefinedOrders.second_first
        else:
            if acronym == "rf":
                return DefinedOrders.second_first
            elif acronym == "knn":
                return DefinedOrders.second_first
            elif acronym == "nb":
                return DefinedOrders.second_first
    elif prediction == "FR" or prediction == "DR":
        return DefinedOrders.first_second
    elif prediction == "RF" or prediction == "RD":
        return DefinedOrders.second_first

#['impute encode normalize rebalance features', 'impute encode rebalance discretize features']
def build_pipeline(features_rebalance_order, discretize_rebalance_order):
    pipelines = []
    if features_rebalance_order == DefinedOrders.first_second:
        pipelines.append("impute encode normalize features rebalance")
    else:
        pipelines.append("impute encode normalize rebalance features")

    pipeline = "impute encode "

    if features_rebalance_order == DefinedOrders.first_second and discretize_rebalance_order == DefinedOrders.second_first:
        pipelines.append(pipeline + "rebalance discretize features")
        pipelines.append(pipeline + "discretize features rebalance")
    else:
        if features_rebalance_order == DefinedOrders.first_second and discretize_rebalance_order == DefinedOrders.first_second:
            pipeline += "discretize features rebalance"
        elif features_rebalance_order == DefinedOrders.second_first and discretize_rebalance_order == DefinedOrders.first_second:
            pipeline += "discretize rebalance features"
        elif features_rebalance_order == DefinedOrders.second_first and discretize_rebalance_order == DefinedOrders.second_first:
            pipeline += "rebalance discretize features"
        pipelines.append(pipeline)

    return pipelines

def pseudo_exhaustive_pipelines():
    pipelines = []

    pipelines.append("impute encode normalize features rebalance")
    pipelines.append("impute encode normalize rebalance features")

    pipeline = "impute encode "
    pipelines.append(pipeline + "rebalance discretize features")
    pipelines.append(pipeline + "discretize rebalance features")
    pipelines.append(pipeline + "discretize features rebalance")

    return pipelines



