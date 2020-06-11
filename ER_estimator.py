import numpy as np
from torch import torch
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import shuffle as shuffle
from sklearn.metrics import r2_score as r2_score_sk
class Closed_form_estimator(RegressorMixin):
    def __init__(self, 
                 *args,
                 eigen_decomposition = False,
                 feature_sparsity = False,
                 elastic_feature_sparsity = False,
                 n_permut = 1000,       
                 lambda_init_value = 1e3,
                 tol = 1e-4,
                 max_iter = 1e3, 
                 torch_optimizer = torch.optim.Adam, 
                 optimizer_params = {"lr": 0.5, "betas":[0.5, 0.9]}, 
                 n_jobs=None, 
                 GPU= True, 
                 is_fitted_ = False,
                 random_state = 0,
                 verbose = False,
                 verbose_step = 1,
                  **kwargs):
        
        self.eigen_decomposition = eigen_decomposition
        self.feature_sparsity = feature_sparsity
        self.elastic_feature_sparsity = elastic_feature_sparsity
        self.torch_optimizer = torch_optimizer
        self.optimizer_params = optimizer_params
        self.tol = tol
        self.max_iter = int(max_iter)
        self.n_jobs = n_jobs
        self.GPU = GPU
        self.random_state = random_state
        self.is_fitted_ = False
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.lambda_init_value = lambda_init_value
    
    def fit(self, X, y, evalset = None, **kwargs):
        """ TODO.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
        TODO
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        if not (self.is_fitted_ and self.warm_start):
            self._create_params(X, y)
            self._params = self._tensor_to_params(self._dic_to_tensor(self.param_init))
            self._optim = self.torch_optimizer(self._params.values(), **self.optimizer_params)
            self.old_loss = 1e16
            self.start_iter = 0
        self._run_optimization(X, y, evalset)
        self.is_fitted_ = True      
        return self
    
    def predict(self, X):
        """ TODO.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
        TODO
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return X.dot(self.coef_)
    
    def _run_optimization(self, X, y, evalset):
        columns = ["Iter", "Loss", "Lambda", "R2_train"]
        if type(evalset) is not type(None):
            X_valid, y_valid = evalset
            columns.append("R2_test")
        if self.verbose:
            print(" / ".join(columns))
            
        constants = self._init_constants(X, y)
        X_true, y_true = (constants["X"], constants["y"])
        intercept_loss = torch.sqrt(((y_true - y_true.mean())**2).sum())
        for e in range(self.start_iter, self.max_iter + self.start_iter): 
            y_pred = self._forward(constants)
            true_loss = self._BKK_criterion( y_true, y_pred)
            loss = true_loss
            loss.backward()
            self._optim.step()
            self._optim.zero_grad()
            detached_loss = self._tensor_to_array(loss)
            if self.verbose and e % self.verbose_step == 0 :
                self._set_coef(constants)
                row = [e,
                    detached_loss,
                    np.exp(self._tensor_to_array(self._params["lambda"])),                                
                    self.score(X, y)]
                
                if type(evalset) is not type(None):
                    row.append(self.score(X_valid, y_valid))
                    
                print(" / ".join(row))
                
            if np.abs(detached_loss - self.old_loss) < self.tol * np.abs(detached_loss):
                self._set_coef(constants)
                self.start_iter = e
                self.old_loss = detached_loss
                return detached_loss
            
            self.old_loss = detached_loss
            
        self._set_coef(constants)  
        self.start_iter = e
        self.old_loss = detached_loss
        return detached_loss
    
    def _forward(self, constants):
        if self.elastic_feature_sparsity:
            mu = torch.sigmoid(self._params["feature_elastic_coef"])
            y_hat = self._inversion_forward( constants, feature_sparsity = False)
            sparse_y_hat = self._inversion_forward( constants, 
                                                                   feature_sparsity = True)
            return (y_hat * mu + (1 - mu) * sparse_y_hat)
        elif self.eigen_decomposition:
            return self._eigen_decomposition_forward(constants)
        else:
            return self._inversion_forward( constants, feature_sparsity = self.feature_sparsity)
    
    def score(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        y_pred = self.predict(X)
        return r2_score_sk(y, y_pred)
    
    def _BKK_criterion(self,  y_true, y_pred):
        return torch.norm(y_true - y_pred)
        
    def _add_sparsity_params(self, p, sparsity_type):
        self.param_init[str(sparsity_type)+"_sparsity_vector"] = np.zeros(p)
        self.param_init[str(sparsity_type)+"_sparsity_coef"] = np.log(1e-1)
        
    def _add_elastic_params(self, elastic_type):
        self.param_init[str(elastic_type)+"_elastic_coef"] = np.zeros(1)
        
    def _sparsify(self, sparsity_type, epsilon = 0.01):
        sparse_vector = self._params[str(sparsity_type)+"_sparsity_vector"]
        sparse_vector = sparse_vector - sparse_vector.mean()
        sparse_vector = sparse_vector * (sparse_vector.var()+epsilon)
        sparse_vector = sparse_vector * torch.exp(self._params[str(sparsity_type)+"_sparsity_coef"])
        return torch.sigmoid(sparse_vector)
        
    def _create_params(self, X, y):
        self.param_init = {"lambda": np.log(self.lambda_init_value)}
        if bool(self.feature_sparsity or self.elastic_feature_sparsity) and not self.eigen_decomposition:
            self._add_sparsity_params(X.shape[1], "feature")
        if self.elastic_feature_sparsity:
            self._add_elastic_params("feature")
        
    def _init_constants(self, X, y):
        if self.eigen_decomposition:
            XTX = X.T.dot(X)
            u, eigen, uT = np.linalg.svd(XTX)
            Xu = X.dot(u)
            utXty = Xu.T.dot(y)
            constants = {"X" : X,
                         "y":y, 
                         "eigen":eigen, 
                         "u" : u, 
                         "Xu": Xu,
                         "utXty" : utXty} 
            return self._dic_to_tensor(constants)
        else:
            XTX = X.T.dot(X)
            XTy = X.T.dot(y)
            constants = {"X" : X,
                         "y":y,
                         "XTX": XTX,
                         "Xty" : XTy}
            return self._dic_to_tensor(constants)
        
    def _inversion_forward(self, constants, feature_sparsity):
        X, y, XTX, XTy = constants.values()
        if feature_sparsity:
            sparse_vector = torch.diag(self._sparsify("feature"))
            sparse_X = X @ sparse_vector
            sparse_XTX = sparse_vector @ XTX @ sparse_vector
            sparse_XTy = sparse_vector @ XTy
            penality = torch.exp(self._params["lambda"]) * torch.diag(torch.ones(XTX.shape[0])).float().cuda()
            inv = torch.inverse(sparse_XTX + penality)
            projection_matrix = sparse_X @ inv
            y_hat = projection_matrix @ sparse_XTy
            return y_hat, permuted_y_hat
        else:
            penality = torch.exp(self._params["lambda"]) * torch.diag(torch.ones(XTX.shape[0])).float().cuda()
            inv = torch.inverse(XTX + penality)
            projection_matrix = X @ inv
            y_hat = projection_matrix @ XTy
            return y_hat
    
    def _eigen_decomposition_forward(self, constants):
        X, y, eigen, u, Xu, utXty = constants.values()
        projection_matrix = Xu * self._get_diag(eigen)
        y_hat = projection_matrix @ utXty
        return y_hat
    
    def _get_diag(self, eigen, epsilon = 1e-8):
        penality = torch.exp(self._params["lambda"])
        return (eigen + penality + epsilon) ** (-1)
    
    def _set_coef(self, constants):
        if self.eigen_decomposition:
            self._eigen_decomposition_coef(constants)
        else:
            self._inversion_coef(constants)
        
    def _eigen_decomposition_coef(self, constants):
        X, y, eigen, u, Xu, utXty = constants.values()
        coef = (u * self._get_diag(eigen)) @ utXty 
        self.coef_ = self._tensor_to_array(coef)
        
    def _inversion_coef( self, constants):
        X, y, XTX, XTy = constants.values()
        feature_sparsity = self.feature_sparsity
        if self.GPU:
            identity = torch.diag(torch.ones(XTX.shape[0])).float().cuda()
        else:
            identity = torch.diag(torch.ones(XTX.shape[0])).float()
        penality = torch.exp(self._params["lambda"]) * identity
        if self.elastic_feature_sparsity:
            mu = torch.sigmoid(self._params["feature_elastic_coef"])
            coef = self._inversion_coef_without_sparsity( penality, XTX, XTy) * mu
            coef += self._inversion_coef_with_sparsity( penality, XTX, XTy) * (1 - mu)
        elif feature_sparsity:
            coef = self._inversion_coef_with_sparsity( penality, XTX, XTy)
        else:
            coef = self._inversion_coef_without_sparsity(penality, XTX, XTy)
        self.coef_ = self._tensor_to_array(coef)
        
    def _inversion_coef_with_sparsity(self, penality, XTX, XTy):
        sparse_vector = torch.diag(self._sparsify("feature"))
        sparse_XTX = sparse_vector @ XTX @ sparse_vector
        sparse_XTy = sparse_vector @ XTy
        inv = torch.inverse(sparse_XTX + penality)
        return sparse_vector @ inv @ sparse_XTy
        
    def _inversion_coef_without_sparsity(self, penality, XTX, XTy):
        inv = torch.inverse(XTX + penality)
        return inv @ XTy
    
    def _dic_to_tensor(self, dic):
        if self.GPU:
            return {key : torch.tensor(value).float().cuda() for key, value in dic.items()}
        else:
            return {key : torch.tensor(value).float() for key, value in dic.items()}
        
    def _tensor_to_array(self, tensor):
        if self.GPU:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    
    def _tensor_to_params(self, tensor_dic):
        return {key : torch.nn.Parameter(value) for key, value in tensor_dic.items()}