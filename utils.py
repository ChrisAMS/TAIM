import pandas as pd
import numpy as np
import os


def load_data(data_path, n_plants, p, resample_rule='10T', n_rows=None):
    """
    data_path: directory where the data is saved.
    n_plants: number of plants to load (K).
    resample: resample rule for data aggregation.
    """
    data = [None] * n_plants
    # test = [None] * n_plants
    for path, _, file_names in os.walk(data_path):
        for i, file_name in enumerate(file_names):
            if i + 1 > n_plants:
                break
            print('File "{}" loaded!'.format(file_name))
            data[i] = pd.read_csv(os.path.join(data_path, file_name),\
                                  index_col=0, names=['85m_speed'], parse_dates=True)
            
            print('Original shape: {}'.format(data[i].shape))

            data[i] = data[i].resample(resample_rule).mean().interpolate(method='time')
            # test[i] = data[i]

            data[i] = data[i]['85m_speed'].values
            nans = 0
            for j in range(data[i].shape[0]):
                if np.isnan(data[i][j]):
                    nans += 1
            print('nans: {}'.format(nans))
            print('Resample data shape: {}'.format(data[i].shape))
            
            if n_rows:
                data[i] = data[i][:n_rows]
                
            print('Final shape: {}'.format(data[i].shape))

    data = np.stack(data, axis=0)
    
    # test = data
    
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
def val_loglhood(theta,Y0,X,flag_print):
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
    if(not(theta.shape[0] == (pv*Kv**2 + Kv**2))):
        print("ERROR: dimensiones theta no coinciden con Y0,X")
        
    #se re-construyen matrices A, CovU a partir de vector theta entregado
    A    = np.reshape(theta[:pv*Kv**2],(Kv*pv,Kv)).swapaxes(0,1)
    CovU = np.reshape(theta[pv*Kv**2:],(Kv,Kv)).swapaxes(0,1)
    
    #se chequea que la matriz CovU sea adecuada (semidefinida positiva)
    eig_val_U = np.linalg.eigvals(CovU)
    flag_sdp  = np.all(eig_val_U >= 0) #valores propios no negativos
    
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