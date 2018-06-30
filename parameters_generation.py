from utils import gibbs_sampling
import scipy.stats as stats
import pickle

PARAMETERS_PATH = '/home/chrisams/Documents/datasets/data_TAIM//samples_2_500.pickle'
DATA_PATH = '/home/chrisams/Documents/datasets/data_TAIM/processed/'
date_start = '2011-05'
date_end = '2011-06'
plant_names = [
    'd05b_2010-06-19_2018-03-05.csv',
    'd01_2009-07-12_2018-01-17.csv',
]

q = stats.norm
K = 2
p = 1
iters = 500
debug = False
mh_iters = 10
n_rows = None # Number of rows of the data to load
method = 'normal'
init_mle = True
annealing = True
T0 = 300
TF = 1
annealing_n = 5
X=None
Y0=None
T = lambda t: T0 * ((TF / T0) ** (t / annealing_n))

samples = gibbs_sampling(iters, DATA_PATH, K, p, q, mh_iters=mh_iters, init_mle=init_mle, n_rows=n_rows,\
                         debug=False, method='normal', X=X, Y0=Y0, annealing=annealing, T=T,\
                         annealing_n=annealing_n, date_start=date_start, date_end=date_end,\
                         plant_names=plant_names)

with open(PARAMETERS_PATH, 'wb') as f:
    pickle.dump(samples, f)