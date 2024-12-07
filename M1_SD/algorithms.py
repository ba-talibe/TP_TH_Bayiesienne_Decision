# %%
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from abc import ABC, abstractmethod

class Modele(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, X, y):
        pass

       
    def plot_decision(self, X, sample = 300):
        """Uses Matplotlib to plot and fill a region with 2 colors
        corresponding to 2 classes, separated by a decision boundary

        Parameters
        ----------
        sample : int, optional
            Number of samples on each feature (default is 300)
        """

        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        x1_list = np.linspace(x1_min, x1_max, sample)
        x2_list = np.linspace(x2_min, x2_max, sample)
        y_grid_pred = [[self.predictor(np.array([x1,x2])) for x1 in x1_list] for x2 in x2_list] 
        plt.contourf(x1_list, x2_list, y_grid_pred, levels=1,alpha=0.35)


    def plot_decision_multi(self, X, sample = 300):
        """Uses Matplotlib to plot and fill a region with 2 colors
        corresponding to 2 classes.

        Parameters
        ----------
        sample : int, optional
            Number of samples on each feature (default is 300)
        """

        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        x1_list = np.linspace(x1_min, x1_max, sample)
        x2_list = np.linspace(x2_min, x2_max, sample)
        y_grid_pred = [[self.predictor (np.array([[x1,x2]]))[0] for x1 in x1_list] for x2 in x2_list] 
        l = np.shape(np.unique(y_grid_pred))[0] - 1
        plt.contourf(x1_list, x2_list, y_grid_pred, levels=l, colors=plt.rcParams['axes.prop_cycle'].by_key()['color'], alpha=0.35)


# %%



class EuclideanClassifier(Modele):
    def __init__(self):
        pass
    
    def fit(self, X, y, k=5):
        classes = np.unique(y)
        means = np.array([X[y == classe].mean(axis=0) for classe in classes])
        
        self.predictor = self._get_predictor(means)
        self.rank_predictor = self._get_rank_predictor(means, k)

    def _get_predictor(self, means):
        def predict(x):
            if len(x.shape) == 1:
                return np.argmin([np.linalg.norm(mean - x) for mean in means]) + 1
            return np.argmin([np.linalg.norm(mean - x, axis=1) for mean in means], axis=0)  + 1
        return predict
    
    def _get_rank_predictor(self, means, k):
        def predict(x):
            if len(x.shape) == 1:
                return (np.argsort([np.linalg.norm(mean - x) for mean in means])[:k]) + 1
            return (np.argsort([np.linalg.norm(mean - x, axis=1) for mean in means], axis=0)[:k]).T + 1
        return predict

class MahalanobisClassifier(Modele):
    def __init__(self):
        self.predictor = None
    
    def fit(self, X, y):
        classes = np.unique(y)
        means = np.array([X[y == classe].mean(axis=0) for classe in classes])
        covs = np.array([np.cov(X[y == classe].T, rowvar=True) for classe in classes])
        inv_covs = np.array([np.linalg.inv(cov) for cov in covs])
        probas = np.array([X[y == classe].shape[0]/X.shape[0] for classe in classes])
        

        dets_sigma = [np.linalg.det(cov) for cov in covs]

        b = [np.log(det_sigma) - 2 * np.log(proba) for  det_sigma, proba in zip(dets_sigma, probas)]
        

        self.predictor = self._get_predictor(means, covs, b)
        self.rank_predictor = self._get_rank_predictor(means, covs, b)

    def _get_predictor(self, means, covs, b, distance=lambda x, y, sigma: np.sum((x@np.linalg.inv(sigma))*y, axis=1)):
        def predict(x):
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)
                return np.argmin([distance(x-mean, x-mean, cov) + b for mean, cov, b in zip(means, covs, b)]) + 1
            return np.argmin([distance(x-mean, x-mean, cov) + b for mean, cov, b in zip(means, covs, b)], axis=0) + 1
        return predict

    def _get_rank_predictor(self, means, covs, b, k=5,distance=lambda x, y, sigma: np.sum((x@np.linalg.inv(sigma))*y, axis=1)):
        def predict(x):
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)
                pred =(np.argsort([distance(x-mean, x-mean, cov) + b for mean, cov, b in zip(means, covs, b)], axis=0)) + 1
                return pred.reshape(-1)[:k]
            return (np.argsort([distance(x-mean, x-mean, cov) + b for mean, cov, b in zip(means, covs, b)], axis=0)[:k]).T + 1
        return predict 
# Function to add labels on top of ba


# %% [markdown]
# Les deux modéles classifie trés bien le jeux de données N° 1.
# 
# Le classifieur de mahalanobis est plus performant que le classifieur euclidien sur le jeu de données 2 de dernier presente des cluster avec un covariance des variable.
# 
# Les deux modéles ont tous du mal à classifier le jeu de données numero 3.
# N'empeche le classifieur de mahalanobis est plus performant.

# %% [markdown]
# ## Parzen

# %%
def gaussian_kernel(x):
    return (1/np.pi**len(x))*np.exp(-x.T@x/2)

def uniform_kernel(x):
    return 1/2*(np.linalg.norm(x) <= .5)

# %%
class Parzen(Modele):
    def __init__(self, kernel=gaussian_kernel, h=1):
        self.kernel = kernel
        self.h = h
        self.predictor = None
        self.rank_predictor = None

    def fit(self, X, y, ):
        self.classes = np.unique(y)
        self.estimators = []
        self.X = X
        self.y = y
        self.probas = np.array([self.X[self.y == classe].shape[0]/self.X.shape[0] for classe in classes])
        self.predictor = self._get_predictor(self.h)
        self.rank_predictor = self._get_rank_predictor(self.h)
    
    def _vector_predict(self, x, h):
        
        estimations = []
        for classe in self.classes:
            XI = self.X[self.y == classe]
            estimation = np.sum([gaussian_kernel((x - xi)/h) for xi in XI])/(len(self.X) * h**self.X.shape[1])
            estimations.append(estimation)
        estimations = np.array(estimations)
        
        probas_a_posteriori = np.array([estimation * proba for estimation, proba in zip(estimations, self.probas)])
        pred = np.argmax(probas_a_posteriori)+1
        return pred
    
    def _vector_rank_predict(self, x, h, k):
        
        estimations = []
        for classe in self.classes:
            XI = self.X[self.y == classe]
            estimation = np.sum([gaussian_kernel((x - xi)/h) for xi in XI])/(len(self.X) * h**self.X.shape[1])
            estimations.append(estimation)
        estimations = np.array(estimations)

        probas_a_posteriori = np.array([estimation * proba for estimation, proba in zip(estimations, self.probas)])
        pred = np.argsort(probas_a_posteriori)[-k:]+1
        return pred

    def _get_predictor(self, h):
        def predict(x):
            if len(x.shape) == 1:
                return self._vector_predict(x, h)
            return np.array([self._vector_predict(xi, h) for xi in x])
        return predict

    def _get_rank_predictor(self, h, k=5):
        def predict(x):
            if len(x.shape) == 1:
                return self._vector_rank_predict(x, h, k)[::-1]
            return np.array([self._vector_rank_predict(xi, h, k)[::-1] for xi in x])
        return predict


class KPP(Modele):
    def __init__(self, k, norm=np.linalg.norm):
        self.k = k
        self.norm = norm

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.dist= np.array([[np.linalg.norm(xi - xj) for xj in X] for xi in X])
        self.predictor = self._get_predictor()
        self.rank_predictor = self._get_rank_predictor()

    def _vector_predict(self, x):
        ppv_classes = self.y[np.argsort(self.dist[np.argmin([self.norm(xi - x) for xi in self.X])])][:self.k]
        return np.argmax(np.bincount(ppv_classes.astype(int)))
    
    def _vector_rank_predict(self, x):
        ppv_classes = self.y[np.argsort(self.dist[np.argmin([self.norm(xi - x) for xi in self.X])])][:self.k]
        return np.argsort(np.bincount(ppv_classes.astype(int)))[::-1]
    
    def _get_predictor(self):
        def predict(x):
            if len(x.shape) == 1:
               return self._vector_predict(x)
            return np.array([self._vector_predict(xi) for xi in x])
        return predict

    def _get_rank_predictor(self, k=5):
        max_len = np.unique(self.y).shape[0]
        def predict(x):
            if len(x.shape) == 1:
               pred = self._vector_rank_predict(x)[:max_len]
               pred = np.pad(pred, (0, max_len - len(pred)), 'constant')
               return pred
            pred = [self._vector_rank_predict(xi)[:max_len] for xi in x]
            pred = [np.pad(d, (0, max_len - len(d)), 'constant') for d in pred]
            return np.array(pred)
        return predict



class KPP(Modele):
    def __init__(self, k, norm=np.linalg.norm, criteria="majority"):
        self.k = k
        self.norm = norm
        if criteria in ["majority", "unanimity"]:
            self.criteria = criteria

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.dist= np.array([[np.linalg.norm(xi - xj) for xj in X] for xi in X])
        self.predictor = self._get_predictor()
        self.rank_predictor = self._get_rank_predictor()

    def _vector_predict(self, x):
        ppv_classes = self.y[np.argsort(self.dist[np.argmin([self.norm(xi - x) for xi in self.X])])][:self.k]
        if self.criteria == "majority":
            return np.argmax(np.bincount(ppv_classes.astype(int)))
        else:
            if np.all(ppv_classes == ppv_classes[0]):
                return ppv_classes[0]
            else:
                return -1
    
    def _vector_rank_predict(self, x):
        ppv_classes = self.y[np.argsort(self.dist[np.argmin([self.norm(xi - x) for xi in self.X])])][:self.k]
        return np.argsort(np.bincount(ppv_classes.astype(int)))[::-1]
    
    def _get_predictor(self):
        def predict(x):
            if len(x.shape) == 1:
               return self._vector_predict(x)
            return np.array([self._vector_predict(xi) for xi in x])
        return predict

    def _get_rank_predictor(self, k=5):
        max_len = np.unique(self.y).shape[0]
        def predict(x):
            if len(x.shape) == 1:
               pred = self._vector_rank_predict(x)[:max_len]
               pred = np.pad(pred, (0, max_len - len(pred)), 'constant')
               return pred
            pred = [self._vector_rank_predict(xi)[:max_len] for xi in x]
            pred = [np.pad(d, (0, max_len - len(d)), 'constant') for d in pred]
            return np.array(pred)
        return predict



class Perceptron(Modele):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        pass

class PerceptronNaif(Perceptron):
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, X, y, maxiter=1000):
        self.classes = np.unique(y)
        X_0 = X[y == self.classes[0]]
        X_1 = X[y == self.classes[1]]

        self.X_t = np.concatenate([X_0, np.ones((X_0.shape[0], 1))], axis=1)
        self.X_t = np.concatenate([self.X_t, np.concatenate([-X_1, -np.ones((X_1.shape[0], 1))], axis=1)], axis=0)
        self.w = np.zeros(self.X_t.shape[1])
        #y_t = np.concatenate([np.ones(X_0.shape[0]), -np.ones(X_1.shape[0])])
        iter = 0
        well_classified = 0
        while well_classified != self.X_t.shape[0] :
            errors=0
            for i in range(self.X_t.shape[0]):
                x = self.X_t[i]
                if x@self.w <= 0:
                    self.w += x
                    well_classified = 0
                    errors += 1 
                else:
                    well_classified += 1

                if well_classified == self.X_t.shape[0]:
                    break
            iter += 1
            if iter == maxiter:
                print("max iter reached")
                break
            
        self.predictor = self._get_predictor()

    def _get_predictor(self):
        def predict(x):
            if len(x.shape) == 1:
                x = np.concatenate([x, [1]])
                return self.classes[0] if x@self.w > 0 else self.classes[1]
            x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
            res =  (x@self.w < 0).astype(int)
            return np.where(res == 0, self.classes[0], self.classes[1])
        return predict



# %%
class LineairSeparator(Modele):
    def __init__(self, perceptronModel : Perceptron, perceptron_kwargs={}, methode="one_vs_one"):
        self.perceptronModel = perceptronModel
        self.perceptron_kwargs = perceptron_kwargs
        if methode in ["one_vs_one", "one_vs_all"]:
            self.methode = methode
        else:
            raise ValueError("methode should be 'one_vs_one' or 'one_vs_all'")

        
    def fit(self, X, y):
        self.classes = np.unique(y).tolist()
        self.perceptrons = []
        if self.methode == "one_vs_one":
            classes_combinations = list(combinations(self.classes, 2))
            for comb in classes_combinations:
                perceptron = self.perceptronModel(**self.perceptron_kwargs)
                X_train = np.concatenate((
                    X[y == comb[0]],
                    X[y == comb[1]]
                ), axis=0) 
                y_train = np.concatenate((
                    y[y == comb[0]],
                    y[y == comb[1]]
                ), axis=0) 
                perceptron.fit(X_train, y_train)
                self.perceptrons.append(perceptron)
        else:
            for classe in self.classes:
                perceptron = self.perceptronModel(**self.perceptron_kwargs)
                y_train = np.where(y == classe, classe, -1)
                perceptron.fit(X, y_train)
                self.perceptrons.append(perceptron)
     
        
        self.predictor = self._get_predictor()


    def _get_predictor(self):
        if self.methode == "one_vs_one":
            def predict(x):
                if len(x.shape) == 1:
                    preds = [perceptron.predictor(x) for perceptron in self.perceptrons]
                    return np.array([np.argmax(np.bincount(preds))])
                preds = [
                    [perceptron.predictor(xi) for perceptron in self.perceptrons] for xi in x]
                preds = np.array(preds).astype(int)
                return np.array([np.argmax(classes) for classes in [np.bincount(pred) for pred in preds]])
        else:
            def predict(x):
                def get_classe(pred):
                        pred = pred[pred != -1]
                        if len(pred) == 0:
                            return 0
                        elif len(pred) > 1:
                            return 0
                        else:
                            return pred[0]
                        
                if len(x.shape) == 1:
                    preds = np.array([perceptron.predictor(x) for perceptron in self.perceptrons])
                    print(preds)
                    return get_classe(preds)
                
                preds = [
                    [perceptron.predictor(xi) for perceptron in self.perceptrons] for xi in x]
                preds = np.array(preds).astype(int)
                preds = np.array([get_classe(pred) for pred in preds])
                return preds
        return predict




class PerceptronEfficient(Perceptron):
    def __init__(self, batch_size=100, epochs=10):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs_on_trainning = 0

    def create_batch(self, X):
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i:i+self.batch_size]

    def fit(self, X, y, verbose=False):
        self.classes = np.unique(y)
        X_0 = X[y == self.classes[0]]
        X_1 = X[y == self.classes[1]]

        self.X_t = np.concatenate([X_0, np.ones((X_0.shape[0], 1))], axis=1)
        self.X_t = np.concatenate([self.X_t, np.concatenate([-X_1, -np.ones((X_1.shape[0], 1))], axis=1)], axis=0)
        self.w = np.zeros(self.X_t.shape[1])

        epoch = 0
        well_classified = 0
        while epoch < self.epochs:
            well_classified = 0
            for batch_X in self.create_batch(self.X_t):
                batch_hyt = batch_X@self.w
                correction = batch_X[batch_hyt <= 0]
                if correction.size != 0:
                    self.w += np.sum(correction, axis=0)
                else:
                    well_classified += len(batch_X) - len(correction)
            
            if well_classified == self.X_t.shape[0]:
                break
            epoch += 1
        if verbose: 
            print(f"epochs {epoch} : {well_classified} well classified")
        self.predictor = self._get_predictor()

    def _get_predictor(self):
        def predict(x):
            if len(x.shape) == 1:
                x = np.concatenate([x, [1]])
                return self.classes[0] if x@self.w > 0 else self.classes[1]
            x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
            res =  (x@self.w < 0).astype(int)
            return np.where(res == 0, self.classes[0], self.classes[1])
        return predict
