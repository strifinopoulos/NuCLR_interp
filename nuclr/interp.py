import sys
sys.path.append(r'C:\Users\gorth\Dropbox (MIT)\Shared\Papers\AI for nuclear\ai-nuclear-nn_test_long_run')
sys.path.append(r'C:\Users\gorth\Dropbox (MIT)\Shared\Papers\AI for nuclear\ai-nuclear-nn_test_long_run\lib')

import torch
import numpy as np
import matplotlib.pyplot as plt
from fitting import preds_targets_zn, get_range_dat, find_local_minima_maxima, envelope, polynomial, PCA_fit, calculate_PCA, get_nucl_range, mask_uncertainities, rms_val
import seaborn as sns
from scipy.signal import find_peaks
import gvar
from nuclr.train import Trainer
from nuclr.data import get_nuclear_data
from data import semi_empirical_mass_formula, WS4_mass_formula, BW_mass_formula
from symbolic import pysr_fit


sns.set()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
path = "..\spiral"
trainer = Trainer.from_path(path, which_folds=[0])

# %%
{k:v**.5 if "metric" in k else v for k,v in trainer.val_step().items()}
# %%

data = trainer.data
# print("Loaded Data:", data._fields, "\n")

model = trainer.models[0]
model.load_state_dict(trainer.models[0].state_dict())

min_included_nucl = trainer.args.INCLUDE_NUCLEI_GT
magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]

#%% PCA FITS

pca_fit_nr = 3
fit_func = envelope
n_fit = 5

Z = np.arange(1,torch.max(data[0][:, 0])+2,1)
N = np.arange(1,torch.max(data[0][:, 1])+2,1)

nucl = Z
embs = model.emb[0].weight.detach().cpu().numpy()

nucl_min = 13
nucl_max = N[-1]
parity = 'all'

Xfit, embs = get_nucl_range(nucl, embs, nucl_min, nucl_max, parity)
PCA_embs, embs = calculate_PCA(embs, False, 5)

# Xfit = np.repeat([embs[:, 0]], 2, axis=0).T

Yfit = PCA_embs[:, pca_fit_nr]
myfit = PCA_fit(Xfit,Yfit,fit_func,n_fit)

model_pysr = pysr_fit(Xfit,Yfit)

best_idx = model_pysr.equations_.query(
    f"loss < {2 * model_pysr.equations_.loss.min()}"
).score.idxmax()
model_pysr.sympy(best_idx)

y_prediction = model_pysr.predict(Xfit.reshape(-1, 1), index=best_idx)

plt.plot(Xfit, PCA_embs[:, pca_fit_nr], "o")
plt.plot(Xfit, y_prediction, "x")

plt.plot(Xfit,gvar.mean(fit_func(Xfit,myfit.p)))
plt.show()
plt.clf()


sys.exit()
# #%%FAST FOURIER ANALYSIS

# pca_fit_nr = 2
# trunc = True

# embs = model.emb[0].weight.detach().cpu().numpy()

# Xfit, embs = get_nucl_range(nucl, embs, nucl_min, nucl_max, parity)
# PCA_embs, embs = calculate_PCA(embs, False, 5)

# Yfourier = np.fft.fft(Yfit)
# Yfit = PCA_embs[:, pca_fit_nr]

# plt.plot(Xfit, PCA_embs[:, pca_fit_nr], "o")

# if trunc:
#     indices_to_keep = [item[0] for item in find_local_minima_maxima(Yfit)]
#     indices_to_keep += [i for i, val in enumerate(Xfit) if val in magic_numbers]
#     indices_to_keep = sorted(indices_to_keep)
    
#     Xfit = Xfit[indices_to_keep]
#     Yfourier = Yfourier[indices_to_keep]

# Yfitinv = np.fft.ifft(Yfourier)

# plt.plot(Xfit, Yfitinv)
# plt.show()
# plt.clf()

#%% CHECK RMS

task_name = "binding_semf"
X, targets, preds = preds_targets_zn(model, data, task_name, train=True, val=True)

mask_unc = mask_uncertainities(8, task_name)
# mask_unc = None

semf_preds = semi_empirical_mass_formula(X[:, 0], X[:, 1]).cpu().numpy()
BW_preds = BW_mass_formula(X[:, 0], X[:, 1]).cpu().numpy()
default_preds = 0*targets
WS4_preds = WS4_mass_formula(get_nuclear_data(False), min_included_nucl)

print(rms_val(X, targets, default_preds, task_name, mask_unc=mask_unc))
print(rms_val(X, targets, BW_preds, task_name, mask_unc=mask_unc))
print(rms_val(X, targets, preds, task_name, mask_unc=mask_unc))
print(rms_val(X, targets, semf_preds, task_name, mask_unc=mask_unc))
print(rms_val(X, targets, WS4_preds, task_name, mask_unc=mask_unc))


#%% REPRODUCING ISOTOPIC CHAINS

task_names = list(data.output_map.keys())   # get a list of names of tasks (e.g. binding_semf)
magic_numbers = [2, 8, 20, 28, 50, 82, 126][1::]
# keep Z fixed and change N see how BE changes.
task_name = "binding_semf"
task_idx = task_names.index(task_name)
Z_range = np.array(20)
N_range = np.arange(20,32,1)

X, targets, preds = preds_targets_zn(model, data, task_name, train=True, val=True)
X_train, targets_train, preds_train = preds_targets_zn(model, data, task_name, train=True, val=False)
X_val, targets_val, preds_val = preds_targets_zn(model, data, task_name, train=False, val=True)

X, targets, preds = get_range_dat(X, targets, preds, Z_range, N_range, clear_nan=False)
X_train, targets_train, _ = get_range_dat(X_train, targets_train, preds_train, Z_range, N_range, clear_nan=True)
X_val, targets_val, _ = get_range_dat(X_val, targets_val, preds_val, Z_range, N_range, clear_nan=True)


#%% PLOTTING FIGURES

# Create the initial lineplot
sns.lineplot(x=X[:, 1], y=preds, label='NuCLR', alpha=0.8)

plt.scatter(x=X_train[:, 1], y=targets_train, label='exp (train)', marker='x', c='black', alpha=1)
plt.scatter(x=X_val[:, 1], y=targets_val, label='exp (val)', marker='x', c='red', alpha=1)

plt.xlabel("N")
plt.legend()
# plt.ylabel(r"$E_B [\rm GeV]$")
plt.ylabel(r"$R_{\rm ch} [\rm fm]$")

plt.title("Calcium")
plt.show()


