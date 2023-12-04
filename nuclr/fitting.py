import numpy as np
import torch
from sklearn.decomposition import PCA
import lsqfit
from data import get_nuclear_data, semi_empirical_mass_formula, BW_mass_formula
from pysr import PySRRegressor

def preds_targets_zn(model, data, task_name, train=True, val=True):
    # the data has an admittedly weird structure
    # data.X is a tensor of shape (N, 3) where N is the number of nuclei
    # TIMES the number of tasks. The first column is the number of protons,
    # the second column is the number of neutrons, and the third column is
    # the task index. 
    task_names = list(data.output_map.keys()) 
    
    if train and val:
        mask = torch.tensor([True for i in range(len(data[0]))])
    elif train:
        mask = data.train_masks[0]
    elif val:
        mask = data.val_masks[0] 
    task_idx = task_names.index(task_name)
    X_train = data.X[mask]
    
    tasks = X_train[:, 2].cpu().numpy()
    scatter = tasks == task_idx # get only rows relevant to task
    X_train_task = X_train[scatter][:,0:2]

    # get the targets and predictions for the task
    # first, we need to undo the preprocessing
    # data.regresion_transformer is a sklearn transformer that does the preprocessing
    # we can use its inverse_transform method to undo the preprocessing
    # it expects a numpy array, of shape (samples, features) where features is the number
    # of tasks we have.
    targets = data.y.view(-1, len(data.output_map.keys())).cpu().numpy()
    targets = data.regression_transformer.inverse_transform(targets)
    targets = targets.flatten()[mask.cpu().numpy()]
    targets = targets[scatter]

    # Predictions on the other hand are shape (samples, tasks)
    # each row has one correct prediction, and the rest are not useful
    # this is not optimal but not worth optimizing for now
    preds = model(data.X[mask])
    preds = preds.cpu().detach().numpy()
    preds = data.regression_transformer.inverse_transform(preds)[scatter, task_idx]
    
    semf = semi_empirical_mass_formula(X_train_task[:, 0], X_train_task[:, 1]).cpu().numpy()
    
    if task_name == 'binding_semf':
        preds = preds + semf
        targets = targets + semf
    
    return X_train_task, targets, preds
   
    
def get_range_dat(X_task, targets, preds, Z_range, N_range, clear_nan=False):
    
    if clear_nan:
        ind = ~np.isnan(targets)
        X_task = X_task[ind]
        targets = targets[ind]
        preds = preds[ind]

    inputs_indices = [i for i,nuclei in enumerate(X_task) if nuclei[0].item() in Z_range and nuclei[1].item() in N_range]
    X_task = X_task[inputs_indices]
    targets = targets[inputs_indices]
    preds = preds[inputs_indices]
    
    return X_task, targets, preds
    
def find_local_minima_maxima(data):
    local_minima_maxima = []

    for i in range(1, len(data) - 2):
        if data[i - 1] < data[i] > data[i + 1]:
            local_minima_maxima.append([i, data[i], "max"])
        elif data[i - 1] > data[i] < data[i + 1]:
            local_minima_maxima.append([i, data[i], "min"])

    return local_minima_maxima

def calculate_PCA(embedding, modified_PCA, n):
    
    # Calculate the PCA components
    pca = PCA(n_components=n)
    pca.fit(embedding)
    PCA_embedding = pca.fit_transform(embedding)

    # Reconstruct the modified embedding using the modified PCA components
    if modified_PCA:
        embedding = pca.inverse_transform(modified_PCA)
    
    print("PCA:", pca.explained_variance_ratio_, "\n")
    return PCA_embedding, embedding

# def get_fit_embeddings(X, ):
    
#     if 
def get_nucl_range(nucl, embs, nucl_min, nucl_max, parity):
    # Extract the first column (nucl values) from the tensor

    # Create a mask based on the specified conditions
    if parity == 'all':
        mask = ((nucl >= nucl_min) & (nucl <= nucl_max)).squeeze()
        
    else:    
        mask = ((nucl >= nucl_min) & (nucl <= nucl_max) & (nucl % 2 == parity)).squeeze()

    # Apply the mask to filter rows
    filtered_embs = embs[mask]
    filtered_nucl = nucl[mask]

    return filtered_nucl, filtered_embs


def envelope(X, p):
  if type(X)==dict:
      [x] = X.values()
  else:
      x = X
  [A, x0, B, f, y0] = p.values()
  fun = A*(X-x0)**2+B*np.sin(f*X)-y0
  return fun

def polynomial(X, p):
  if type(X)==dict:
      [x] = X.values()
  else:
      x = X
  [a] = p.values()
  fun = 0
  for i in range(len(a)):
      fun += a[i]*(x)**i
  return fun  
    
def PCA_fit(X, y, fit_func, n_pol):
    
  magic_numbers = [2, 8, 20, 28, 50, 82, 126]
  # prior = {"a": gvar.gvar([0.065], [0.1])}      
  
  if fit_func == polynomial:
      p0 = {"a": [0.4]*(n_pol+1)}
  elif fit_func == envelope:                 
      p0 = {"A": [2*10**(-4)], "x0": [70], "B": [0.35], "f":[0.2], "y0":[0.8]}
  
  shape = y.shape[0]
  cov = np.zeros((shape,shape), dtype=int)
  for i in range(shape):
     if X[i] in magic_numbers:
         cov[i, i] = 4
     else:
         cov[i, i] = 1
  return lsqfit.nonlinear_fit(data=(X, y, cov), fcn=fit_func, p0=p0, svdcut=1e-12)

def mask_uncertainities(min_included_nucl, task_name):
    df = get_nuclear_data(False)
    
    df = df[
        (df.z > min_included_nucl) & (df.n > min_included_nucl)
    ]
    
    if task_name=="binding_semf":
        mask = np.logical_or((df.binding_unc * (df.z + df.n) < 100).values,~(df.binding_sys == 'Y').values)
    elif task_name=="radius":
        mask = (df.unc_r < 0.005).values
    return mask


def rms_val(X, targets, preds, task_name, mask_unc=None):  
    
    # returns the value of rms in keV
    
    A = np.array(X[:,0]+X[:,1])
    mask = ~np.logical_or(np.isnan(targets),np.isnan(preds))
    
    if mask_unc is not None:
        mask = np.logical_and(mask,mask_unc)
    
    targets = targets[mask]
    preds = preds[mask]    
    A = A[mask]
    
    return np.sqrt(np.mean((A*(targets-preds))**2))
