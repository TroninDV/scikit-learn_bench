# ===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import argparse

import bench
import daal4py
import numpy as np
import catboost as cb
from catboost_converter import get_gbt_model_from_catboost


def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])


def convert_cb_predictions(y_pred, objective):
    if objective == 'multi:softprob':
        y_pred = convert_probs_to_classes(y_pred)
    elif objective == 'Logloss':
        y_pred = y_pred.astype(np.int32)
    return y_pred


parser = argparse.ArgumentParser(
    description='catboost gbt + model transform + daal predict benchmark')


# Not found
parser.add_argument('--colsample-bytree', type=float, default=1,
                    help='Subsample ratio of columns '
                         'when constructing each tree')


parser.add_argument('--count-pool', default=False, action='store_true',
                    help='Count Pool creation in time measurements')


parser.add_argument('--grow-policy', type=str, default='Depthwise',
                    help='Controls a way new nodes are added to the tree')

# Not found
parser.add_argument('--inplace-predict', default=False, action='store_true',
                    help='Perform inplace_predict instead of default')


parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                    help='Step size shrinkage used in update '
                         'to prevents overfitting')

parser.add_argument('--max-bin', type=int, default=256,
                    help='Maximum number of discrete bins to '
                         'bucket continuous features')
# Not found
parser.add_argument('--max-delta-step', type=float, default=0,
                    help='Maximum delta step we allow each leaf output to be')

parser.add_argument('--max-depth', type=int, default=6,
                    help='Maximum depth of a tree')

parser.add_argument('--max-leaves', type=int, default=0,
                    help='Maximum number of nodes to be added')

# Not found
parser.add_argument('--min-child-weight', type=float, default=1,
                    help='Minimum sum of instance weight needed in a child')

# Not found
parser.add_argument('--min-split-loss', '--gamma', type=float, default=0,
                    help='Minimum loss reduction required to make'
                         ' partition on a leaf node')

parser.add_argument('--n-estimators', type=int, default=100,
                    help='Number of gradient boosted trees')

parser.add_argument('--objective', type=str, required=True,
                    choices=('RMSE', 'Logloss',
                             'multi:softmax', 'multi:softprob'),
                    help='Control a balance of positive and negative weights')

# choices=('reg:squarederror', 'binary:logistic',
#          'multi:softmax', 'multi:softprob'),

# Not found
parser.add_argument('--reg-alpha', type=float, default=0,
                    help='L1 regularization term on weights')

parser.add_argument('--reg-lambda', type=float, default=1,
                    help='L2 regularization term on weights')

parser.add_argument('--scale-pos-weight', type=float, default=1,
                    help='Controls a balance of positive and negative weights')

parser.add_argument('--subsample', type=float, default=1,
                    help='Subsample ratio of the training instances')

params = bench.parse_args(parser)

X_train, X_test, y_train, y_test = bench.load_data(params)

cb_params = {
    'verbose': 0,
    'learning_rate': params.learning_rate,
    # 'min_split_loss': params.min_split_loss,
    'max_depth': params.max_depth,
    # 'min_child_weight': params.min_child_weight,
    # 'max_delta_step': params.max_delta_step,
    'subsample': params.subsample,
    # 'sampling_method': 'uniform',
    # 'colsample_bytree': params.colsample_bytree,
    'colsample_bylevel': 1,  # Only one available
    # 'colsample_bynode': 1,
    'reg_lambda': params.reg_lambda,
    # 'reg_alpha': params.reg_alpha,
    # 'tree_method': params.tree_method,    
    'grow_policy': params.grow_policy,
    'max_bin': params.max_bin,
    'objective': params.objective,
    'random_seed': params.seed,
    'iterations': params.n_estimators
}

if cb_params['grow_policy'] == 'Lossguide':
    cb_params['max_leaves'] = params.max_leaves

if params.threads != -1:
    cb_params.update({'thread_count': params.threads})

if params.objective == "RMSE":
    task = 'regression'
    metric_name, metric_func = 'rmse', bench.rmse_score
else:
    task = 'classification'
    metric_name = 'accuracy'
    metric_func = bench.accuracy_score
    if 'cudf' in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))

    # Covtype has one class more than there is in train
    if params.dataset_name == 'covtype':
        params.n_classes += 1
    
    if params.n_classes > 2:        
        cb_params['bootstrap_type'] = 'Bernoulli'
        cb_params['classes_count'] = params.n_classes
    else:        
        cb_params['scale_pos_weight'] = params.scale_pos_weight

    
    

t_creat_train, dtrain = bench.measure_function_time(cb.Pool, X_train, params=params,
                                                    label=y_train)
t_creat_test, dtest = bench.measure_function_time(
    cb.Pool, X_test,  params=params, label=y_test)


def fit(pool):
    if pool is None:
        pool = cb.Pool(X_train, label=y_train)
    return cb.CatBoost(cb_params).fit(pool)
    # return cb.train(cb_params, pool, params.n_estimators)


if cb_params['objective'].startswith('multi'):
    if cb_params['objective'] == 'multi:softmax':
        def predict(pool):
            if pool is None:
                pool = cb.Pool(X_test, label=y_test)
            return booster.predict(pool, prediction_type='Class')
    else:
        def predict(pool):
            if pool is None:
                pool = cb.Pool(X_test, label=y_test)
            return booster.predict(pool, prediction_type='Probability')
    cb_params['objective'] = 'MultiClass'
else:    
    if cb_params['objective'] == 'Logloss':
        def predict(pool):
            if pool is None:
                pool = cb.Pool(X_test, label=y_test)
            return booster.predict(pool, prediction_type = 'Class')
    else:
        def predict(pool):
            if pool is None:
                pool = cb.Pool(X_test, label=y_test)
            return booster.predict(pool)


fit_time, booster = bench.measure_function_time(
    fit, None if params.count_pool else dtrain, params=params)


train_metric = metric_func(
    convert_cb_predictions(predict(dtrain), params.objective),
    y_train)

predict_time, y_pred = bench.measure_function_time(
    predict, None if params.count_pool else dtest, params=params)

test_metric = metric_func(convert_cb_predictions(y_pred, params.objective), y_test)

transform_time, model_daal = bench.measure_function_time(
    get_gbt_model_from_catboost, booster, params=params)

if hasattr(params, 'n_classes'):
    predict_algo = daal4py.gbt_classification_prediction(
        nClasses=params.n_classes, resultsToEvaluate='computeClassLabels', fptype='float')
    predict_time_daal, daal_pred = bench.measure_function_time(
        predict_algo.compute, X_test, model_daal, params=params)
    test_metric_daal = metric_func(y_test, daal_pred.prediction)
else:
    predict_algo = daal4py.gbt_regression_prediction()
    predict_time_daal, daal_pred = bench.measure_function_time(
        predict_algo.compute, X_test, model_daal, params=params)
    test_metric_daal = metric_func(y_test, daal_pred.prediction)

bench.print_output(
    library='modelbuilders', algorithm=f'catboost_{task}_and_modelbuilder',
    stages=['training_preparation', 'training', 'prediction_preparation', 'prediction',
            'transformation', 'alternative_prediction'],
    params=params,
    functions=['catboost.Pool', 'Catboost.train', 'xgb.dmatrix.test', 'xgb.predict',
               'daal4py.get_gbt_model_from_xgboost', 'daal4py.compute'],
    times=[t_creat_train, fit_time, t_creat_test, predict_time, transform_time,
           predict_time_daal],
    accuracy_type=metric_name,
    accuracies=[None, train_metric, None, test_metric, None, test_metric_daal],
    data=[X_train, X_train, X_test, X_test, X_test, X_test])
