from pandas import DataFrame

from predictor import FlightDelayPredictor

from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              )
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor

models_param_grid = [
    (LinearRegression, {}),
    (Lasso, {"alpha": [x * 0.1 for x in range(10, 50)]}),

    (Ridge, {"alpha": [x * 0.1 for x in range(10, 50)]}),

    (ElasticNet, {"alpha": [x * 0.1 for x in range(10, 50)], 'l1_ratio': [x * 0.1 for x in range(1, 10)], }),

    (AdaBoostRegressor, {'n_estimators': [50, 100, 500, 1000, 1500], 'learning_rate': [x * 0.1 for x in range(1, 10)],
                         "loss": ['linear', 'square', 'exponential'], }),

    (GradientBoostingRegressor,
     {'n_estimators': [50, 100, 500, 1000, 1500], 'learning_rate': [x * 0.1 for x in range(1, 10)]}),

    (MLPRegressor,
     {'learning_rate': ['constant', 'invscaling', 'adaptive'], 'activation': ['identity', 'logistic', 'tanh', 'relu'],
      "alpha": [x * 0.0001 for x in range(10, 50)], "solver": ['lbfgs', 'sgd', 'adam'],
      'hidden_layer_sizes': [50, 100, 200, ]},),

    (SVR, {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'degree': [x for x in range(1, 10)],
           'C': [1, 2, 3, 4], 'epsilon': [x * 0.1 for x in range(1, 10)]})

]
flight_delay_predictor = FlightDelayPredictor('./flight_delay.csv')

if __name__ == '__main__':
    """Visualise Data"""
    flight_delay_predictor.visualize()

    """ Remove Outliers"""
    flight_delay_predictor.remove_outliers()

    """Visualise Effect of Outlier Removal"""
    flight_delay_predictor.visualize()

    """Cross Validate And Get Best Hyper Parameters"""
    best_params = []
    for model_class, parameter_grid in models_param_grid:
        best_params.append(flight_delay_predictor.get_best_parameters(model_class, parameter_grid))

    models = [model for model, parameter_grid in models_param_grid]
    models_best_params = list(zip(models, best_params))

    """Train Models"""
    trained_models = []
    for model, best_params in models_best_params:
        trained_models.append(flight_delay_predictor.train_model(model, **best_params))

    """ Evaluate with Training Set """
    training_evaluation = []
    for model in trained_models:
        scores = flight_delay_predictor.evaluate_model(model, flight_delay_predictor.X_train,
                                                       flight_delay_predictor.Y_train)
        training_evaluation.append({"model": model, **scores})

    print(DataFrame(training_evaluation))

    """ Evaluate with Test Set"""
    evaluation = []
    for model in trained_models:
        scores = flight_delay_predictor.evaluate_model(model)
        evaluation.append({"model": model, **scores})

    print(DataFrame(evaluation))
