from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer, KBinsDiscretizer, \
    Binarizer, OneHotEncoder, OrdinalEncoder

from .utils import generate_domain_space

import pandas as pd
import numpy as np


class PrototypeSingleton:
   __instance = None

   POOL = {
       "impute": [None, SimpleImputer(), IterativeImputer()],
       "encode": [None, OneHotEncoder()],
       #"encode": [None, OneHotEncoder(), OrdinalEncoder()],
       "rebalance": [None, NearMiss(), SMOTE()],
       #"rebalance": [None, NearMiss(), CondensedNearestNeighbour(), SMOTE()],
       "normalize": [None, StandardScaler(), PowerTransformer(), MinMaxScaler(), RobustScaler()],
       "discretize": [None, KBinsDiscretizer(), Binarizer()],
       "features": [None, PCA(), SelectKBest(), FeatureUnion([("pca", PCA()), ("selectkbest", SelectKBest())])]
   }

   PROTOTYPE = {}
   DOMAIN_SPACE = {}
   parts = []
   X = []
   y = []
   original_numerical_features = []
   original_categorical_features = []
   current_numerical_features = []
   current_categorical_features = []


   @staticmethod
   def getInstance():
      """ Static access method. """
      if PrototypeSingleton.__instance == None:
         PrototypeSingleton()
      return PrototypeSingleton.__instance

   def __init__(self):
      """ Virtually private constructor. """
      if PrototypeSingleton.__instance != None:
         raise Exception("This class is a singleton!")
      else:
         PrototypeSingleton.__instance = self

   def setPipeline(self, params):
       for param in params:
           self.parts.append(param)

       for part in self.parts:
           self.PROTOTYPE[part] = self.POOL[part]

       self.DOMAIN_SPACE = generate_domain_space(self.PROTOTYPE)

   def setFeatures(self, num_features, cat_features):
       self.original_numerical_features = num_features
       self.current_numerical_features = num_features
       self.original_categorical_features = cat_features
       self.current_categorical_features = cat_features

   def set_X_y(self, X, y):
       self.X = X
       self.y = y

   def resetFeatures(self):
       self.current_numerical_features = self.original_numerical_features
       self.current_categorical_features = []
       self.current_categorical_features.extend(self.original_categorical_features)

   def applyColumnTransformer(self):
       len_numerical_features = len(self.current_numerical_features)
       len_categorical_features = len(self.current_categorical_features)
       self.current_numerical_features = list(range(0, len_numerical_features))
       self.current_categorical_features = list(range(len_numerical_features, len_categorical_features + len_numerical_features))

   def applyDiscretization(self):
       self.current_categorical_features.extend(self.current_numerical_features)
       self.current_numerical_features = []

   def applyOneHotEncoding(self):
       new_categorical_features = 0
       for i in self.original_categorical_features:
           new_categorical_features += len(np.unique(self.X[:, i]))
       old_categorical_features = len(self.original_categorical_features)
       old_features = len(self.original_categorical_features) + len(self.original_numerical_features)
       new_features = old_features - old_categorical_features + new_categorical_features

       len_numerical_features = len(self.current_numerical_features)
       self.current_numerical_features = list(range(0, len_numerical_features))
       self.current_categorical_features = list(range(len_numerical_features, new_features))

   def getFeatures(self):
       return self.current_numerical_features, self.current_categorical_features

   def getDomainSpace(self):
       return self.DOMAIN_SPACE

   def getPrototype(self):
       return self.PROTOTYPE

   def getParts(self):
       return self.parts