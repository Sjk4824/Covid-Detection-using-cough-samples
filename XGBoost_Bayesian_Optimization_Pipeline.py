
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import hp, tpe, fmin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import plot_roc_curve, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from pathlib import Path
import index_helpers as ih
import data_transformations as dtrans


segmentation = True
fine_segmentation = True
exlude_expert=False
exclude_meta_data=False


df, _, _, _ = ih.read_and_merge_data(segmentation, fine_segmentation, exlude_expert, exclude_meta_data)
if segmentation:
    df, count = ih.index_df_by_person(df)
else:
    df = df.set_index(['File_Name'])
df = dtrans.low_var_exclusion(df, 0.1)

df = df[df["Expert"]!=2]

df = pd.get_dummies(df, columns=['Resp_Condition', 'Gender', 'Expert'])

if segmentation:
    X_train, X_test, y_train, y_test = ih.train_test_split_on_index(features = df.drop("Label", axis=1),
                                                            label = df["Label"])
    groups = y_train.reset_index()['File_Name_split']
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop("Label", axis=1), df["Label"], test_size=0.2)
    groups = y_train.reset_index()['File_Name']

X_t = X_train.reset_index(drop=False)
y_t = y_train.reset_index(drop=False)


from imblearn.over_sampling import RandomOverSampler
RUS = RandomOverSampler(random_state=42)
X_res, y_res = RUS.fit_resample(X_t, y_t["Label"])

df_res = X_res.merge(y_res, left_index=True, right_index=True)
if segmentation:
    df_res.set_index(['File_Name_split', 'File_n_recording'])
else:
    df_res.set_index(['File_Name'])

    
if segmentation:
    X = df_res.drop(columns=['File_Name_split', 'File_n_recording', 'Label'])
    y = df_res['Label']
    groups = df_res["File_Name_split"]
else:
    X = df_res.drop(columns=['File_Name', 'Label'])
    y = df_res['Label']
    groups = df_res["File_Name"]
    
    
param_hyperopt = {'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1)),
                  'max_depth': hp.quniform('max_depth', 20, 100, 5),
                  'max_delta_step': hp.quniform('max_delta_step', 0, 20, 1),
                  'gamma': hp.uniform ('gamma', 1,9),
                  'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
                  'reg_lambda' : hp.uniform('reg_lambda', 0,1),
                  'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
                  'min_child_weight' : hp.quniform('min_child_weight', 0, 20, 1),
                  'n_estimators': hp.quniform('n_estimators', 50, 300, 10)}

def objective(params):
    
    params = {'learning_rate': float(params['learning_rate']),
              'max_depth': int(params['max_depth']),
              'max_delta_step': int(params['max_delta_step']),
              'gamma': int(params['gamma']),
              'reg_alpha': int(params['reg_alpha']),
              'reg_lambda': float(params['reg_lambda']),
              'colsample_bytree': float(params['colsample_bytree']),
              'min_child_weight': int(params['min_child_weight']),
              'n_estimators': int(params['n_estimators'])}
    
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic',**params)
    
    gkf=GroupKFold(n_splits=5)
    best_score = cross_val_score(xgb_clf, X, y, cv=gkf, groups=groups, 
                                 scoring='roc_auc', n_jobs=-1).mean()
    
    return -best_score
    
best_result = fmin(fn=objective, space=param_hyperopt, max_evals=25, algo=tpe.suggest, rstate=np.random.RandomState(42))
best_result


best_result_cast = {'learning_rate': float(best_result['learning_rate']),
                  'max_depth': int(best_result['max_depth']),
                    'max_delta_step': int(best_result['max_delta_step']),
                  'gamma': int(best_result['gamma']),
                  'reg_alpha': int(best_result['reg_alpha']),
                  'reg_lambda': float(best_result['reg_lambda']),
                  'colsample_bytree': float(best_result['colsample_bytree']),
                  'min_child_weight': int(best_result['min_child_weight']),
                  'n_estimators': int(best_result['n_estimators'])}


best_clf = xgb.XGBClassifier(objective='binary:logistic', **best_result_cast)

best_clf.fit(X, y)

from sklearn.metrics import roc_auc_score
pred=best_clf.predict(X_test)
roc_auc_score(pred,y_test)

X_v = X_test.reset_index(drop=True)
y_v = y_test.reset_index(drop=True)
preds = best_clf.predict_proba(X_v)
plot_roc_curve(best_clf, X_v, y_v)

import shap
explainer = shap.KernelExplainer(best_clf.predict_proba, shap.kmeans(X_train, 50), link="logit")
shap_values = explainer.shap_values(X_v, nsamples=100)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], X_v)
