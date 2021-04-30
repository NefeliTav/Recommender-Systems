import os
from surprise import SVD

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import *
from surprise.prediction_algorithms import *


file_path = os.path.expanduser('./dataset/ratings_1.csv')
reader = Reader(line_format='user item rating', sep=',',
                rating_scale=[1, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

########################################################################

algo = NormalPredictor()
cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

bsl = {
    'method': "sgd",  # Optimization method to use.
    # Learning rate parameter for the SGD optimization method.
    'learning_rate': 0.005,
    'n_epochs': 50,  # The number of iteration for the SGD optimization method.
    # The regularization parameter of the cost function that is optimized: a.k.a. LAMBDA.
    'reg': 0.02,
}
algo = BaselineOnly(bsl_options=bsl, verbose=True)
cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

max_neighbors = 40
min_neighbors = 1
# A dictionary of options for the similarity measure
similarity_options = {
    'user_based': False,  # True ==> UserUser-CF, False ==> ItemItem-CF
    'name': "cosine",  # The name of the similarity measure to use.
    'min_support': 3,
    # The minimum number of common items/users for the similarity not to be zero.
}

algo = KNNBasic(k=max_neighbors, min_k=min_neighbors,
                sim_options=similarity_options, verbose=True)

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

algo = KNNWithMeans(k=max_neighbors, min_k=min_neighbors,
                    sim_options=similarity_options, verbose=True)

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

algo = KNNWithZScore(k=max_neighbors, min_k=min_neighbors,
                     sim_options=similarity_options, verbose=True)

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

algo = KNNBaseline(k=max_neighbors, min_k=min_neighbors,
                   sim_options=similarity_options, verbose=True)

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

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

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

algo = SVDpp(n_factors=number_of_factors,
             n_epochs=number_of_epochs,
             lr_all=learning_rate,
             reg_all=lambda_parameter,
             verbose=True)

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

########################################################################

algo = NMF(n_factors=number_of_factors,
           biased=use_together_with_baseline_estimator,
           n_epochs=number_of_epochs,
           lr_all=learning_rate,
           reg_all=lambda_parameter,
           verbose=True)

cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)
