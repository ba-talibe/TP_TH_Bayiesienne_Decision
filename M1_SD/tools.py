from algorithms import *

def top_k_accuracy(X, y_true, modele : Modele, k=2):
    if k == 1:
        labels = modele.predictor(X)
        score = [ 1 if y_true[i] == labels[i] else 0 for i in range(len(y_true))]
        
    else:
        labels = modele.rank_predictor(X)
        score = [ 1 if y_true[i] in labels[i,:k] else 0 for i in range(len(y_true))]

    return np.mean(score)


def validate(valid_data, modele : Modele, name=None, top_k_metrics=False):
    
    x_valid, y_valid = valid_data[["x1", "x2"]].to_numpy(), valid_data.y.to_numpy()
    classes = np.unique(y_valid)
    
    y_pred = modele.predictor(x_valid)
    valid_data["y_pred"] = y_pred

    top1, top2 = None, None
    if top_k_metrics:
        top1 = top_k_accuracy(x_valid, y_valid, modele, 1)
        print(f"top1 : {top1}")
        if "rank_predictor" in dir(modele):
            top2 = top_k_accuracy(x_valid, y_valid, modele, 2)
            print(f"top2 : {top2}")
        

    matrice_confusion = np.zeros((len(classes), len(classes)))
    matrice_confusion = pd.DataFrame(matrice_confusion)
    matrice_confusion.columns =  np.arange(len(classes))
    matrice_confusion.index = matrice_confusion.columns
    
    for classe1 in  range(len(classes)):
        for classe2 in  range(len(classes)):
            matrice_confusion.loc[classe1, classe2] = valid_data[(valid_data.y_pred == classe1+1) & (valid_data.y == classe2+1)].shape[0]


    if len(classes) == 2:
        plt.figure()

        plt.axis('off')
        plt.show()

        plt.figure()
        
        plt.text(0, 0, f"vrai positive : {matrice_confusion.loc[0, 0]}", ha='center', fontsize=12, va='center', color='red')
        plt.text(1, 0, f"faux positive : {matrice_confusion.loc[0, 1]}", ha='center', fontsize=12,  va='center', color='red')
        plt.text(0, 1, f"faux negative : {matrice_confusion.loc[1, 0]}", ha='center', fontsize=12, va='center', color='red')
        plt.text(1, 1, f"vrai negative : {matrice_confusion.loc[1, 1]}", ha='center', fontsize=12, va='center', color='red')

        plt.axis("off")
        plt.imshow(matrice_confusion, cmap='YlGnBu', interpolation='nearest')
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrice_confusion, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
        plt.title(f"Matrice de confusion {name}")
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
    
    return [top1, top2]

def show(df, save=True, title=None, delta =None):
    
    classes = df.y.unique()
    marker =  ["+","x",".","o",",","*","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
    repartitons = {classe : df[df.y == classe] for classe in classes}
    
    plt.figure(figsize=(10, 8))
    
    for i, classe in enumerate(classes):
        card = repartitons[classe].size
        plt.scatter(repartitons[classe].x1, repartitons[classe].x2, marker=marker[i], alpha=.7, label=f"{str(classe)}, {card}")


    if title is not None:
        plt.title(title)

    if delta is not None:
        for deltai in delta:
            plt.axvline(deltai )
            plt.text(deltai, 2, f"{np.round(deltai)}")

    plt.axis("equal")
    plt.legend()

def plot_decision(x1_min, x1_max, x2_min, x2_max, prediction, sample = 300):
    """Uses Matplotlib to plot and fill a region with 2 colors
    corresponding to 2 classes, separated by a decision boundary

    Parameters
    ----------
    x1_min : float
        Minimum value for the first feature
    x1_max : float
        Maximum value for the first feature
    x2_min : float
        Minimum value for the second feature
    x2_max : float
        Maximum value for the second feature
    prediction :  (x : 2D vector) -> label : int
        Prediction function for decision
    sample : int, optional
        Number of samples on each feature (default is 300)
    """
    x1_list = np.linspace(x1_min, x1_max, sample)
    x2_list = np.linspace(x2_min, x2_max, sample)
    y_grid_pred = [[prediction(np.array([x1,x2])) for x1 in x1_list] for x2 in x2_list] 
    plt.contourf(x1_list, x2_list, y_grid_pred, levels=1,alpha=0.35)

def plot_decision_multi(x1_min, x1_max, x2_min, x2_max, prediction, sample = 300):
    """Uses Matplotlib to plot and fill a region with 2 colors
    corresponding to 2 classes.

    Parameters
    ----------
    x1_min : float
        Minimum value for the first feature
    x1_max : float
        Maximum value for the first feature
    x2_min : float
        Minimum value for the second feature
    x2_max : float
        Maximum value for the second feature
    prediction :  (x : 2D vector) -> label : int
        Prediction function for a vector x
    sample : int, optional
        Number of samples on each feature (default is 300)
    """
    x1_list = np.linspace(x1_min, x1_max, sample)
    x2_list = np.linspace(x2_min, x2_max, sample)
    y_grid_pred = [[prediction(np.array([[x1,x2]]))[0] for x1 in x1_list] for x2 in x2_list] 
    l = np.shape(np.unique(y_grid_pred))[0] - 1
    plt.contourf(x1_list, x2_list, y_grid_pred, levels=l, colors=plt.rcParams['axes.prop_cycle'].by_key()['color'], alpha=0.35)

def execute_modele(modele : Modele, data=None, sample=300, top_k_metrics=False):
    if data is not None:
        X_train = data[["x1", "x2"]].to_numpy()
        y_train = data.y.to_numpy().astype(int)
        modele.fit(X_train, y_train)
        show(data)
        modele.plot_decision_multi(X_train, sample=sample)
        top_results = validate(valid_data[data], modele, top_k_metrics=top_k_metrics)
        plt.show()
    else:   
        top_results = []
        for data in datas:
            X_train = datas[data][["x1", "x2"]].to_numpy()
            y_train = datas[data].y.to_numpy()
            X_valid = valid_data[data][["x1", "x2"]].to_numpy()
            y_valid = valid_data[data].y.to_numpy()
            modele.fit(X_train, y_train)
            show(valid_data[data], title=f"Train Data {data[-1]}")
            modele.plot_decision_multi(X_valid, sample=sample)
            top_result = validate(valid_data[data], modele, name=f"Validation Data {data[-1]}", top_k_metrics=top_k_metrics)
            top_results.append(top_result)
            plt.show()
    return top_results

def cross_validation(modele : Modele, X, y, cv=5):
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    indices = np.array_split(indices, cv)
    top1_scores = []
    top2_scores = []
    for i in range(cv):
        test_indices = indices[i]
        train_indices = np.concatenate([indices[j] for j in range(cv) if j != i])
        modele.fit(X[train_indices], y[train_indices])
        top1_score = top_k_accuracy(X[test_indices], y[test_indices], modele, 1)
        top2_score = top_k_accuracy(X[test_indices], y[test_indices], modele, 2)
        top1_scores.append(top1_score)
        top2_scores.append(top2_score)
    return np.mean(top1_scores), np.mean(top2_scores)



def plot_linear_separator(model, X, y):
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
    
    # Create a mesh grid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict the class for each point in the mesh grid
   
    Z = model.predictor(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Linear Separator')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
