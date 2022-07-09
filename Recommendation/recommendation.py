import os
import time
from surprise import SVD

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import *
from surprise.prediction_algorithms import *

file_path = os.path.expanduser('../dataset/ratings_1.csv')
reader = Reader(line_format='user item rating', sep=',',
                rating_scale=[1, 5], skip_lines=1)
data1 = Dataset.load_from_file(file_path, reader=reader)
file_path = os.path.expanduser('../dataset/ratings_2.csv')
reader = Reader(line_format='user item rating', sep=',',
                rating_scale=[1, 10], skip_lines=1)
data2 = Dataset.load_from_file(file_path, reader=reader)
jobs = 8  # number of cores

# part 1
for data in [data1, data2]:  # for both datasets, try all prediction algorithms
    # Basic Algorithms
    # normal distribution of the training set
    algo = NormalPredictor()
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # baseline estimate for given user and item
    algo = BaselineOnly(bsl_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # k-NN Algorithms
    # basic collaborative filtering algorithm
    algo = KNNBasic(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # taking into account the mean ratings of each user
    algo = KNNWithMeans(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # taking into account the z-score normalization of each user
    algo = KNNWithZScore(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # taking into account a baseline rating
    algo = KNNBaseline(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # Matrix Factorization
    number_of_factors = 100  # The number of factors.
    use_together_with_baseline_estimator = True  # Whether to use baselines.
    # The number of iterations for the SGD optimization method.
    number_of_epochs = 20
    # Learning rate parameter for the SGD optimization method.
    learning_rate = .005
    # The regularization parameter of the cost function that is optimized: a.k.a. LAMBDA.
    lambda_parameter = .02

    algo = SVD(n_factors=number_of_factors,
               biased=use_together_with_baseline_estimator,
               n_epochs=number_of_epochs,
               lr_all=learning_rate,
               reg_all=lambda_parameter,
               verbose=True)

    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # Probabilistic Matrix Factorization
    number_of_factors = 100  # The number of factors.
    use_together_with_baseline_estimator = False  # Whether to use baselines.
    # The number of iterations for the SGD optimization method.
    number_of_epochs = 20
    # Learning rate parameter for the SGD optimization method.
    learning_rate = .005
    # The regularization parameter of the cost function that is optimized: a.k.a. LAMBDA.
    lambda_parameter = .02

    algo = SVD(n_factors=number_of_factors,
               biased=use_together_with_baseline_estimator,
               n_epochs=number_of_epochs,
               lr_all=learning_rate,
               reg_all=lambda_parameter,
               verbose=True)

    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # taking into account implicit ratings
    number_of_factors = 20
    number_of_epochs = 20
    learning_rate = .007
    lambda_parameter = .02

    algo = SVDpp(n_factors=number_of_factors,
                 n_epochs=number_of_epochs,
                 lr_all=learning_rate,
                 reg_all=lambda_parameter,
                 verbose=True)

    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # Non-negative Matrix Factorization
    number_of_factors = 15
    number_of_epochs = 50
    use_together_with_baseline_estimator = False
    learning_rate = .007

    algo = NMF(n_factors=number_of_factors,
               biased=use_together_with_baseline_estimator,
               n_epochs=number_of_epochs,
               verbose=True)
    print()
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # SlopeOne algorithm
    algo = SlopeOne()
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    # co-clustering
    algo = CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=20,
                        random_state=None, verbose=False)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print()
    print()

# part 2
for data in [data1, data2]:  # for both datasets ,try different configurations of knnbaseline, SVD

    param_grid = {'k': [10, 20, 30, 40, 50, 55, 60, 70, 80, 90], 'min_k': [1, 2, 3, 5, 10, 15],
                  'sim_options':
                  {'name': ['cosine', 'pearson_baseline', 'msd', 'pearson'], 'min_support': [1, 5, 10],
                   'user_based': [True, False], 'shrinkage': [0, 50, 100, 200, 300]},
                  'bsl_options':
                  {'method': ['als', 'sgd'], 'n_epochs': [1, 2, 5, 10, 20, 30]
                   }
                  }

    gs = RandomizedSearchCV(KNNBaseline, param_grid, measures=[
                            'rmse', 'mae'], cv=5, n_jobs=jobs)

    gs.fit(data)
    print(gs.best_score['rmse'])  # best RMSE score
    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    #start_time = time.time()

    param_grid = {'n_epochs': [40, 42, 43], 'lr_all': [
        0.0055, 0.006, 0.0065, 0.007], 'reg_all': [0.10, 0.15, 0.2]}
    gs = GridSearchCV(SVD, param_grid, measures=[
                      'rmse', 'mae'], cv=5, n_jobs=jobs)

    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    #end_time = time.time()
    #print("Completed in:", end_time-start_time, "seconds")
