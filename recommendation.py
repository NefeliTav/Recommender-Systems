import os
import time
from multiprocessing import Pool
from surprise import SVD

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import *
from surprise.prediction_algorithms import *

file_path = os.path.expanduser('./dataset/ratings_1.csv')
reader = Reader(line_format='user item rating', sep=',',
                rating_scale=[1, 5], skip_lines=1)
data1 = Dataset.load_from_file(file_path, reader=reader)
file_path = os.path.expanduser('./dataset/ratings_2.csv')
reader = Reader(line_format='user item rating', sep=',',
                rating_scale=[1, 10], skip_lines=1)
data2 = Dataset.load_from_file(file_path, reader=reader)
jobs = 8  # number of cores

print("First Dataset\n")
for data in [data1, data2]:
    # Basic Algorithms
    # normal distribution of the training set
    algo = NormalPredictor()
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")

    # baseline estimate for given user and item
    algo = BaselineOnly(bsl_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")

    # k-NN Algorithms
    # basic collaborative filtering algorithm
    algo = KNNBasic(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")

    # taking into account the mean ratings of each user
    algo = KNNWithMeans(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")

    # taking into account the z-score normalization of each user
    algo = KNNWithZScore(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")

    # taking into account a baseline rating
    algo = KNNBaseline(k=40, min_k=1, sim_options={}, verbose=True)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")
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

    print("########################################################################")

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

    print("########################################################################")

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

    print("########################################################################")

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

    print("########################################################################")

    # SlopeOne algorithm
    algo = SlopeOne()
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")

    # co-clustering
    algo = CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=20,
                        random_state=None, verbose=False)
    cross_validate(algo, data, measures=[
        'RMSE', 'MAE'], cv=5, n_jobs=jobs, verbose=True)

    print("########################################################################")
    print()
    print()
    print("Second Dataset\n")
