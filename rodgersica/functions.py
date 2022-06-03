import numpy as np

#input Jacobian, K, [n x m], error covariance matrix Se, [m x m] and a priori matrix Sa, [n x n]
#jac_me and me are associated with model uncertainty - that parameterized uncertainty and its jacobian
def rodgers(jac, err, ap, model_error={}, model_error_jacobian={}): 
    '''calculate the parameter error covariance matrix'''
    
    #todo
    # consider microplastic simulation jacobian in the following manner: Se' = Se + Kb Sb Kbt, where 
    #  Se is the same as above, Sb is the microplastic parameter uncertainty, Kb the microplastic jacobian
    
        #check if error covariance matrix is square, or just diagonal values. If latter make full matrix
    if err.ndim == 1:
        ln=np.shape(err)
        err2d = np.zeros((ln[0], ln[0]))
        np.fill_diagonal(err2d, err)
        err=err2d

        #check if a priori covariance matrix is square, or just diagonal values. If latter make full matrix
    if ap.ndim == 1:
        ln=np.shape(ap)
        ap2d = np.zeros((ln[0], ln[0]))
        np.fill_diagonal(ap2d, ap)
        ap=ap2d        
            
        #section to verify compatable dimensions ------------------------------------------------------
    sh_jac = np.shape(jac)
    sh_err = np.shape(err)
    sh_ap = np.shape(ap)
    
    n_dim = sh_jac[0]
    m_dim = sh_jac[1]
    
    if not((sh_err[0] == sh_err[1]) and (sh_ap[0] == sh_ap[1])):
        print('ERROR: error covariance matrix or a priori matrix are not square')
        print('Error covariance matrix dimensions')
        print(sh_err)
        print('A priori matrix dimensions')
        print(sh_ap)
        return -1, -1, -1, -1
    
    if not(sh_jac[0] == sh_ap[0]):
        print('ERROR: n dimensions inconsistent, should be Jacobian [n x m]; a priori [n x n]')
        print('Jacobian matrix dimensions')
        print(sh_jac)
        print('A priori matrix dimensions')
        print(sh_ap)
        return -1, -1, -1, -1
    
    if not(sh_jac[1] == sh_err[0]):
        print('ERROR: m dimensions inconsistent, should be Jacobian [n x m]; error covariance [m x m]')
        print('Jacobian matrix dimensions')
        print(sh_jac)
        print('Error covariance matrix dimensions')
        print(sh_err)
        return -1, -1, -1, -1
        
    #section to generate model derived error -------------------------------------------------------
    
    if len(model_error) > 0:
        me=model_error
        jac_me=model_error_jacobian
        
        ln_me=np.shape(me)
        errme_2d = np.zeros((ln_me[0], ln_me[0]))
        np.fill_diagonal(errme_2d, me)
        err_me=errme_2d
    
        jac_me_t=np.transpose(jac_me)      
    
        JacmetMeJacme = np.matmul(jac_me_t,np.matmul(err_me,jac_me))
        err = err + JacmetMeJacme
    
        #perform inverse and matrix multiplication calculations ----------------------------------------
    jac_t=np.transpose(jac) #transpose of Jacobian (KT)
    
    try: 
        err_i=np.linalg.inv(err) #inverse of error covariance matrix (Se-1)
    except:
        print("ERROR: problem inverting error covariance matrix")
        return -1, -1, -1, -1
    
    try: 
        ap_i=np.linalg.inv(ap) #inverse of a priori error covariance matrix
    except:
        print("ERROR: problem inverting a priori covariance matrix")
        return -1, -1, -1, -1

    KtSK = np.matmul(jac,np.matmul(err_i,jac_t)) #calcuates KT Se-1 K

    try: 
        S_hat = np.linalg.inv(KtSK+ap_i) #calculate the inverse of (above + Sa-1)
    except:
        print("ERROR: problem inverting retrieval error covariance matrix")
        return -1, -1, -1, -1
    
    SIC = 0.5*np.log(np.linalg.det(np.matmul((KtSK+ap_i),ap))) #calculate Shannon Information Content    
    AvgK = np.matmul(S_hat,KtSK) #averaging kernel
    DFS = np.trace(AvgK) #degrees of freedom for signal (DFS) which is trace of averaging kernel
    
    return S_hat, SIC, AvgK, DFS  #returns retrieval error covariance matrix and the Shannon Information Content