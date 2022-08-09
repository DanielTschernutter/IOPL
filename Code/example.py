import helper as hp
from IOPL import IOPL
from sklearn.svm import SVR

## Set parameters

scenario_id = 3 # scenario_id can be set to 1, 2 or 3 (1=basic scenario, 2=complex scenario, 3=very complex scenario)
M = 3 # maximal number of hyperboxes
num_of_samples = 10000
num_datapoints_for_training = 1000

## Estimator for mu

mu_1 = SVR()
mu_minus1 = SVR()
param_grid_mu = {'kernel': ['poly','rbf'],
                 'degree': [2,3,4],
                 'C': [1e-2,1e-1,1.0,1e1,1e2]}

## Get data

X, T, Y = hp.get_simulated_data(scenario_id,num_of_samples=num_of_samples)
X_Train = X[:num_datapoints_for_training,:]
T_Train = T[:num_datapoints_for_training]
Y_Train = Y[:num_datapoints_for_training]

## Training

IOPL_instance = IOPL(X_Train, T_Train, Y_Train, M, mu_1=mu_1, mu_minus1=mu_minus1, param_grid_mu=param_grid_mu)
IOPL_instance.fit()

## Plotting

hp.plot_results(IOPL_instance,scenario_id,X)