from utils import gibbs_sampling
import scipy.stats as stats
import pickle

DATA_PATH = 'C:/Users/Christian/Documents/datasets/TAIM/processed'
PARAMETERS_PATH = 'C:/Users/Christian/Documents/datasets/TAIM/list_gen_params.pickle'
q = stats.norm
K = 3
p = 1
iters = 1
debug = False
mh_iters = 10
n_rows = 10000 # Number of rows of the data to load
method = 'normal'
init_mle = False

samples = gibbs_sampling(iters, DATA_PATH, K, p, q, mh_iters=mh_iters, init_mle=init_mle, n_rows=n_rows, debug=debug, method=method)

with open(PARAMETERS_PATH, 'wb') as f:
    pickle.dump(samples, f)