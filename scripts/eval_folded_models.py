# %%
import os
import torch
from nuclr.train import Trainer

# %%
path = "..\spiral"
trainer = Trainer.from_path(path, which_folds=[0])

# %%
{k:v**.5 if "metric" in k else v for k,v in trainer.val_step().items()}
# %%
trainer.model(0)