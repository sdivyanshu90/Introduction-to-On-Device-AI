# %% [markdown]
# Source notebook: Notebook_1.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %% [markdown]
# # L2: Deploying Segmentation Models On-Device

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %% [markdown]
# [FFNet Paper](https://arxiv.org/abs/2206.08236)

# %%
# Notebook cell 4
from qai_hub_models.models.ffnet_40s import Model

# %%
# Notebook cell 5
from torchinfo import summary

# %%
# Notebook cell 6
# Load from pre-trained weights
model = Model.from_pretrained()
input_shape = (1, 3, 1024, 2048)
stats = summary(model, 
  input_size=input_shape, 
  col_names=["num_params", "mult_adds"]
)
print(stats)

# %% [markdown]
# ## Exercise: Try another variant of FFNet

# %%
# Notebook cell 8
# High resolution variants
from qai_hub_models.models.ffnet_40s import Model
#from qai_hub_models.models.ffnet_54s import Model
#from qai_hub_models.models.ffnet_78s import Model

# Low resolution variants
low_res_input_shape = (1, 3, 512, 1024)
#from qai_hub_models.models.ffnet_78s_lowres import Model
#from qai_hub_models.models.ffnet_122ns_lowres import Model

model = Model.from_pretrained()
stats = summary(model, 
  input_size=input_shape, # use low_res_input_shape for low_res models
  col_names=["num_params", "mult_adds"]
)
print(stats)

# %% [markdown]
# ## Setup AI Hub for device-in-the-loop deployment

# %%
# Notebook cell 10
import qai_hub

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access Utils File and Helper Functions:</b> To access the files for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %%
# Notebook cell 12
from utils import get_ai_hub_api_token
ai_hub_api_token = get_ai_hub_api_token()

get_ipython().system('qai-hub configure --api_token $ai_hub_api_token')

# %%
# Notebook cell 13
get_ipython().run_line_magic('run', '-m qai_hub_models.models.ffnet_40s.demo')

# %% [markdown]
# ## Run on a real smart phone!

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note:</b> To spread the load across various devices, we are selecting a random device. Feel free to change it to any other device you prefer.</p>

# %%
# Notebook cell 16
devices = [
    "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy Tab S8",
    "Xiaomi 12",
    "Xiaomi 12 Pro",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24+",
]

import random
selected_device = random.choice(devices)
print(selected_device)

# %%
# Notebook cell 17
get_ipython().run_line_magic('run', '-m qai_hub_models.models.ffnet_40s.export -- --device "$selected_device"')

# %% [markdown]
# <p style="background-color:#fff1d7; padding:15px; "> <b>Note</b>: To view the URL for each job, you require login. You can experience sample results in the following urls.</p>
# 
# * [FFNet 40s](https://aihub.qualcomm.com/mobile/models/ffnet_40s)
# * [FFNet 54s](https://aihub.qualcomm.com/mobile/models/ffnet_54s)
# * [FFNet 78s](https://aihub.qualcomm.com/mobile/models/ffnet_78s)
# * [FFNet 78s-low-res](https://aihub.qualcomm.com/mobile/models/ffnet_78s_lowres)
# * [FFNet 122ns-low-res](https://aihub.qualcomm.com/mobile/models/ffnet_122ns_lowres)

# %% [markdown]
# ## On Device Demo

# %%
# Notebook cell 20
get_ipython().run_line_magic('run', '-m qai_hub_models.models.ffnet_40s.demo -- --device "$selected_device" --on-device')

# %%
# Notebook cell 21


# %%
# Notebook cell 22


# %%
# Notebook cell 23


# %%
# Notebook cell 24


# %%
# Notebook cell 25

