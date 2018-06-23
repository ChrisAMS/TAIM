import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import truncnorm


def load_data(data_path, n_plants, p, resample_rule='10T', n_rows=None):
    """
    data_path: directory where the data is saved.
    n_plants: number of plants to load (K).
    resample: resample rule for data aggregation.
    """
    data = [None] * n_plants
    for path, _, file_names in os.walk(data_path):
        for i in range(len(file_names)):
            if i + 1 > n_plants:
                break
            data[i] = pd.read_csv(os.path.join(data_path, 'plant_{}.csv'.format(i)),\
                                  index_col=0, names=['85m_speed'], parse_dates=True)
            data[i] = data[i].resample(resample_rule).mean().interpolate(method='time')
            data[i] = data[i]['85m_speed'].values
            if n_rows:
                data[i] = data[i][:n_rows]
    data = np.stack(data, axis=0)
    if p > 0:
        X = np.zeros((n_plants * p, data.shape[1] - p))
        j = 0
        for i in range(p, data.shape[1]):
            for t in range(p):
                X[t * n_plants:(t + 1) * n_plants, j] = data[:, (i - 1) - t]
            j += 1
    else:
        X = data
    data = data[:, p:]
    return data, X


## Calculo matrices Y0,X para loglikehood
def calc_Y0X(data): #data must have mean 0
    #Y0 is computed
    Y0 = data[:,p:] #pre-sample removed for Y0
    
    #X is computed
    X = []
    for i in range(p-1,-1,-1):
        X.append(data[:,i:(N-p+i)]) 
    X = np.concatenate(X,axis=0)
    
    return {"Y0":Y0,"X":X}


## Calculo loglikehood (matricial)
def loglhood(CovU,Y0,A,X):
    #Loglikehood accoring to Lutkepohl 2005
    #Constant term KT/2 ln2pi neglected
    #T = Y0.shape[1]
    
    trace_mat = np.transpose(Y0 - A@X) @ np.linalg.inv(CovU) @ (Y0 - A@X)
    out = -(Y0.shape[1]/2)*np.log(np.linalg.det(CovU)) -1/2*np.trace(trace_mat) 
    return out 


## Calculo loglikehood (element-wise)
def val_loglhood(theta,Y0,X,flag_print, method='normal', init_params=False):
    #theta: vector con coeficientes para A, CovU
    #flag_print: si se desea imprimir A, CovU para su revision
    #A     = [a_1,...,a_Kp], donde a_j corresponde a la columna j de A (j=1,...,Kp)
    #CovU  = [u_1,...,u_K],  donde u_j corresponde a la columna j de CovU (j=1,...,K)
    #theta = [vec(A);vec(CovU)] (; indica abajo, no a la derecha, i.e. formato MATLAB)
    #theta = [a_1;...;a_Kp;u_1;...;u_K]
    
    #dimensiones se pueden rescatar de Y0,X
    Kv = Y0.shape[0] #K
    Tv = Y0.shape[1] #T
    pv = int(X.shape[0]/Kv) #p (orden VAR)
    
    #se verifica que theta tenga las dimensiones correctas
    if pv == 1 and method == 'personalized' and not init_params:
        if(not(theta.shape[0] == (Kv*Kv*2+Kv*2))):
            print("ERROR: dimensiones theta no coinciden con Y0,X")
    else:
        if(not(theta.shape[0] == (pv*Kv**2 + Kv**2))):
            print("ERROR: dimensiones theta no coinciden con Y0,X")

    #se re-construyen matrices A, CovU a partir de vector theta entregado
    if pv == 1 and method == 'personalized' and not init_params:
        A, CovU = reconstruct_coefs(theta, Kv)
    else:
        A    = np.reshape(theta[:pv*Kv**2],(Kv*pv,Kv)).swapaxes(0,1)
        CovU = np.reshape(theta[pv*Kv**2:],(Kv,Kv)).swapaxes(0,1)
        CovU = np.dot(CovU.T,CovU)
    
    
    #se chequea que la matriz CovU sea adecuada (semidefinida positiva)
    eig_val_U = np.linalg.eigvals(CovU)
    flag_sdp  = np.all(eig_val_U >= 0) and np.all(np.isreal(eig_val_U)) #valores propios no negativos y reales 
    
    #se chequea que la matriz A sea adecuada (proceso estable, pag 15 Lutkepohl)
    if(pv==1): #no es necesario agregar bloque 
        A_test = A
    else:
        A_block  = np.block(np.eye(Kv*(pv-1)))
        A_zeros  = np.zeros((Kv*(pv-1),Kv))
        A_bottom = np.concatenate((A_block,A_zeros),axis=1)
        A_test   = np.concatenate((A,A_bottom),axis=0)
    eig_val_A   = np.absolute(np.linalg.eigvals(A_test))
    flag_stable = np.all(eig_val_A < 1) #valores propios absolutos menores a 1
    
    #se evalua la funcion de loglikelihood
    if(not(flag_sdp)):
        val = -np.inf #fuera del soporte
        if(flag_print): #detalles del error
            print("Matriz CovU no es semidefinita positiva")
            print(CovU)
            print(eig_val_U)
    elif(not(flag_stable)):
        val = -np.inf
        if(flag_print): 
            print("Matriz A no es estable")
            print(A)
            print(eig_val_A)
    else: #Parametros matrices A,CovU validos
        val = loglhood(CovU,Y0,A,X) 
        if(flag_print): #se muestran matrices A, CovU construidas
            print("Matriz A resulante:")
            print(A)
            print("-----")
            print("Matriz CovU resulante:")
            print(CovU)
        
    return val


def gibbs_sampling(iters, data_path, K, p, q, mh_iters=1, n_rows=None, debug=False, method='normal'):
    """
    iters: quantity of samples of A and U.
    data_path: path where data is saved.
    K: number of plants (n_plants in load_data function).
    p: past time to be considered.
    q: jumping distribution for parameters (from scipy.stats).
    mh_iters: haw many samples do with Metropolis Hastings.
    n_rows: how many rows of the data to consider.
    debug: debug mode.
    method: normal - use a jumping distribution from scipy.stats
            personalized - use the jumping distribution personalized by us.
    """
    print('Loading data...')
    Y0, X = load_data(data_path, K, p, resample_rule='10T', n_rows=n_rows)
    if debug:
        print('Y0 shape: {}'.format(Y0.shape))
        print('X shape: {}'.format(X.shape))

    # Theta is the vector of all parameters that will be sampled.
    # A and CovU are reshaped to a 1-D vector theta.
    # Note that this theta change dimensionality when using personalized.
    print('Initializing parameters...')
    theta = init_parameters(K, p, q, Y0, X, debug=debug, method=method)
    
    print('Calculating MLE...')
    f = lambda theta: -val_loglhood(theta,Y0,X,False, method=method, init_params=False)
    result = minimize(f, theta)
    theta = result.x
    print('Init MLE theta calculated! ({})'.format(result.fun))

    if debug:
        print('Parameters intialized!')
    samples = []
    for i in range(iters):
        start_it = time.time()
        print('Iteration {}'.format(i))

        # Loop over all parameters and for each parameter theta[j],
        # do a MH sampling over the distribution of theta[j] given theta[-j].
        for j in range(theta.shape[0]):
            start = time.time()
            mh_samples = metropolis_hastings(theta, j, q, mh_iters, Y0, X, K, debug, method=method)
            end = time.time()
            print('Time for sampling theta[{}]: {}'.format(j, end - start))
            # When mh_iters > 1, mh_samples contain mh_iters samples, so a random
            # choice (uniform) is done for selection of the new theta.
            theta[j] = np.random.choice(mh_samples)
        
        lk = val_loglhood(theta,Y0,X,False, method=method, init_params=False)
        print('LK of new theta: {}'.format(lk))

        if p == 1 and method == 'personalized':
            A, CovU = reconstruct_coefs(theta, K)
        else:
            A    = np.reshape(theta[:p*K**2],(K*p,K)).swapaxes(0,1)
            CovU = np.reshape(theta[p*K**2:],(K,K)).swapaxes(0,1)
            CovU = np.dot(CovU.T,CovU)

        samples.append([A, CovU])
        end_it = time.time()
        print('Time for iteration {}: {}'.format(i, end_it - start_it))
    print('Finished!')
    return samples


def metropolis_hastings(theta, j, q, iters, Y0, X, K, debug, method='normal'):
    """
    theta: theta vector with all parameters.
    j: theta index of the parameter currently been sampled.
    q: jumping distribution.
    """
    user_std = 1
    samples_mh = [theta[j]] # start sample.
    lk_old = val_loglhood(theta, Y0, X, debug, method=method)
    # print('init lk: {}'.format(lk_old))
    for t in range(iters):
        lk_new = -np.inf
        c = -1
        while lk_new == -np.inf:
            c += 1
            if method == 'normal':
                x_new = q.rvs(loc=samples_mh[-1], scale=1)
                theta[j] = x_new
            elif method == 'personalized':
                theta, q_eval_new, q_eval_old = jump_dst(theta, j, user_std, K)
            lk_new = val_loglhood(theta, Y0, X, debug, method=method)
            # print('new_lk: {}'.format(lk_new))
        #print('Quantity of -np.infs: {}'.format(c))
        if method == 'normal':
            logalpha = min([lk_new - lk_old + np.log(q.pdf(samples_mh[-1], loc=x_new) \
                                                     / q.pdf(x_new, loc=samples_mh[-1])), 0])
        elif method == 'personalized':
            logalpha = min([lk_new - lk_old + np.log(q_eval_old / q_eval_new), 0])
        alpha = np.exp(logalpha)
        u = stats.uniform.rvs()
        if u < alpha:
            #print('acepted')
            samples_mh.append(theta[j])
            lk_old = lk_new
        else:
            #print('rejected')
            samples_mh.append(samples_mh[-1])
            theta[j] = samples_mh[-1]
    return np.array(samples_mh)
        
    
def init_parameters(K, p, q, Y0, X, method='normal', debug=False):
    """
    Initialization of parameters. This functions search a matrix A
    and a matrix CovU that satisfy some conditions that A and CovU
    must satisfy.
    """
    if debug:
        print('Initializing parameters...')
    while True:
        theta = np.zeros(K ** 2 * (p + 1))
        for i in range(theta.shape[0]):
            theta[i] = q.rvs()

        # Force CovU to be positive semidefinite.
        covu = np.reshape(theta[-K**2:], (K, K)).T
        covu = np.dot(covu.T, covu)
        theta[-K**2:] = np.reshape(covu, K**2)
        
        lk = val_loglhood(theta, Y0, X, debug, method=method, init_params=True)
        if debug:
            print('LK = {}'.format(lk))
        if lk != -np.inf:
            print('lk init: {}'.format(lk))
            if p == 1 and method == 'personalized':
                A = np.reshape(theta[:p*K**2],(K*p,K)).swapaxes(0,1)
                eig_valuesA, eig_vecA = np.linalg.eig(A)
                eig_valuesB, eig_vecB = np.linalg.eig(covu)
                theta = np.concatenate((eig_vecA.reshape(-1), eig_vecB.reshape(-1),
                                        eig_valuesA, eig_valuesB))
                if np.all(np.isreal(eig_valuesA)):
                    break
            else:
                break
    return theta

#DISCLAIMER: CODED FOR VAR OF ORDER 1

## Jumping distribution of theta, conditioned on all values except index j
def jump_dst(theta_old,j,user_std,K):
    #theta_old: previous value of vector theta
    #j: index for which dist is unconditioned
    #user_std: size of step of jumping distribution
    
    dt = 0.0001 #avoid exactly taking limits of bounds

    mu = theta_old[j]
    theta = theta_old.copy()

    # q_eval_new is q(x_new | x_old).
    # q_eval_old is q(x_old | x_new).

    if (j < (K*K*2)):
        # rv = norm(loc=mu,scale=user_std)
        theta[j] = norm.rvs(loc=mu, scale=user_std)
        q_eval_new = norm.pdf(theta[j], loc=mu, scale=user_std)
        q_eval_old = norm.pdf(mu, loc=theta[j], scale=user_std)
    elif ( (j >= (K*K*2)) and (j < (K*K*2+K)) ):
        # a, b = (-1+dt - mu) / user_std, (1-dt - mu) / user_std
        # rv = truncnorm(a=a,b=b,loc=mu,scale=user_std) #bounded between (-1,1)
        a_new, b_new = (-1+dt - mu) / user_std, (1-dt - mu) / user_std
        theta[j] = truncnorm.rvs(a=a_new, b=b_new, loc=mu, scale=user_std)
        a_old, b_old = (-1+dt - theta[j]) / user_std, (1-dt - theta[j]) / user_std
        q_eval_new = truncnorm.pdf(a=a_new, b=b_new, loc=mu, scale=user_std)
        q_eval_old = truncnorm.pdf(a=a_old, b=b_old, loc=theta[j], scale=user_std)
    elif ( (j >= (K*K*2+K)) and (j < (K*K*2+K*2)) ):
        # a  = (0+dt - mu) / user_std
        # rv = truncnorm(a=a,b=np.inf,loc=mu,scale=user_std) #bounded between (0,+inf)
        a_new = (0+dt - mu) / user_std
        theta[j] = truncnorm(a=a_new, b=np.inf, loc=mu, scale=user_std)
        a_old = (0+dt - theta[j]) / user_std
        q_eval_new = truncnorm.pdf(a=a_new, b=np.inf, loc=mu, scale=user_std)
        q_eval_old = truncnorm.pdf(a=a_old, b=np.inf, loc=theta[j], scale=user_std)
    else:
        print("ERROR: index j out of bounds")

    # theta = theta_old.copy()
    # theta[j] = rv.rvs()
    # q_eval = rv.pdf(theta[j])

    # samp_vecA = np.reshape(theta[:(K*K)],(K,K))
    # samp_vecU = np.reshape(theta[(K*K):(K*K*2)],(K,K))
    # samp_valA = np.diag(theta[(K*K*2):(K*K*2+K)])
    # samp_valU = np.diag(theta[(K*K*2+K):(K*K*2+K*2)])

    # A = samp_vecA @ samp_valA @np.linalg.inv(samp_vecA)
    # U = samp_vecU @ samp_valU @np.linalg.inv(samp_vecU)
    
    return(theta, q_eval_new, q_eval_old)


def reconstruct_coefs(theta, K):
    samp_vecA = np.reshape(theta[:(K*K)],(K,K))
    samp_vecU = np.reshape(theta[(K*K):(K*K*2)],(K,K))
    samp_valA = np.diag(theta[(K*K*2):(K*K*2+K)])
    samp_valU = np.diag(theta[(K*K*2+K):(K*K*2+K*2)])

    A = samp_vecA @ samp_valA @np.linalg.inv(samp_vecA)
    U = samp_vecU @ samp_valU @np.linalg.inv(samp_vecU)
    return A, U


def sim_wind(A,CovU,x0,horizon,n_samples):
    #Simula trayectorias de viento
    #A: Matriz de coeficientes de acuerdo a Lutkepohl
    #CovU: Matriz de covarianza ruido U de acuerdo a Lutkhepol
    #x0: puntos de partida a partir del cual se genera el pronostico
    #horizon: horizonte de tiempo hasta el cual se genera el pronostico
    #n_samples: numero de trayectorias a generar
    #x[t] = A_1 x[t-1] + ... + A_p x[t-p] + u[t]
    #A = [A_1,...,A_p]
    #u[t] Normal(0,CovU)
    #x0 = [x[t-1];...;x[t-p]] (; indica abajo, no a la derecha, i.e. formato MATLAB)
    #Formato salida: lista xt[t], donde t corresponde al t-step ahead forecast
    #Cada componente de la lista xt[t] almacena una matriz de dimension (K x n_samples)
    
    #Dimensiones son obtenidas a partir de matrices A, CovU
    Kv = CovU.shape[0] #K (dimension x, i.e. numero centrales)
    pv = int(A.shape[1]/Kv) #p (orden modelo VAR(p))
    
    #Se chequea consistencia con dimensiones x0 (K x p)
    flag_x0 = (x0.shape[0]==(Kv*pv)) and (x0.shape[1]==1)
    if(not(flag_x0)):
        print("ERROR: Las dimensiones de x0 no son consistentes con A,CovU")
        
    #Se chequea horizonte > 0
    if(horizon<1):
        print("ERROR: El horizonte debe ser mayor a 0")
     
    #Simulacion iterativa para todo el horizonte
    xt = np.zeros((Kv,n_samples,horizon))
    x_prev = np.repeat(x0,n_samples,axis=1) #xt de tiempo/iteracion anterior
    #xt_old = []
    for t in range(horizon):
        #generacion ruido aleatorio
        samples = np.random.multivariate_normal(np.zeros(Kv),CovU,size=n_samples)
        
        #modelo VAR(p)
        calc_xt = (A @ x_prev) + np.transpose(samples)
        xt[:,:,t] = calc_xt
        #xt_old.append(calc_xt)
        
        #actualiza x_prev
        x_prev = x_prev[:(Kv*(pv-1)),:]
        x_prev = np.concatenate((calc_xt,x_prev))
        
    return xt

def plot_series(xt):
    #Recibe lista xt con pronosticos generados de sim_wind y
    #reordena los datos. Finalmente grafica los resultados.
    
    dim_series = xt.shape[0] #dimension de series de tiempo
    n_samples  = xt.shape[1] #numero de trayectorias generadas
    horizon    = xt.shape[2] #horizonte pronostico
    
    f, axarr = plt.subplots(dim_series,sharex=True)
    for i in range(dim_series):
        for k in range(n_samples):
            axarr[i].plot(xt[i,k,:])
            axarr[i].set_ylabel('Wind Speed')
            axarr[i].set_xlabel('Time')
    plt.subplots_adjust(left=None, bottom=None, right=3, top=2,
                wspace=None, hspace=0.25)
    plt.show()

def PL5(u,a,b,c,d,g):    
    val = d + (a-d)/((1+(u/c)**b)**g)
    return val


def power_curve(u,cap_wind,cut_speed,a,b,c,d,g):
    #negative values and values over cut out speed are equal to zero
    val = np.zeros(u.shape)          #output array
    filt_speed = (u>0)&(u<cut_speed) #wind speed for which power curve is different from zero
    val[filt_speed] = PL5(u[filt_speed],a,b,c,d,g)
    val = cap_wind*val
    return val