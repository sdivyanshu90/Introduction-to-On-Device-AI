# %% [markdown]
# Source notebook: Notebook_2.ipynb
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
# # L3: Preparing for on-device deployment

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %% [markdown]
# ## Capture trained model

# %%
# Notebook cell 4
from qai_hub_models.models.ffnet_40s import Model as FFNet_40s

# Load from pre-trained weights
ffnet_40s = FFNet_40s.from_pretrained()

# %%
# Notebook cell 5
import torch
input_shape = (1, 3, 1024, 2048)
example_inputs = torch.rand(input_shape)

# %%
# Notebook cell 6
traced_model = torch.jit.trace(ffnet_40s, example_inputs)

# %%
# Notebook cell 7
traced_model

# %% [markdown]
# ## Compile for device

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access Utils File and Helper Functions:</b> To access the files for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %%
# Notebook cell 10
import qai_hub
import qai_hub_models

from utils import get_ai_hub_api_token
ai_hub_api_token = get_ai_hub_api_token()

get_ipython().system('qai-hub configure --api_token $ai_hub_api_token')

# %%
# Notebook cell 11
for device in qai_hub.get_devices():
    print(device.name)

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note:</b> To spread the load across various devices, we are selecting a random device. Feel free to change it to any other device you prefer.</p>

# %%
# Notebook cell 13
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
# Notebook cell 14
device = qai_hub.Device(selected_device)

# Compile for target device
compile_job = qai_hub.submit_compile_job(
    model=traced_model,                        # Traced PyTorch model
    input_specs={"image": input_shape},        # Input specification
    device=device,                             # Device
)

# %%
# Notebook cell 15
# Download and save the target model for use on-device
target_model = compile_job.get_target_model()

# %% [markdown]
# ## Exercise: Try different runtimes 

# %%
# Notebook cell 17
compile_options="--target_runtime tflite"                  # Uses TensorFlow Lite
compile_options="--target_runtime onnx"                    # Uses ONNX runtime
compile_options="--target_runtime qnn_lib_aarch64_android" # Runs with Qualcomm AI Engine

compile_job_expt = qai_hub.submit_compile_job(
    model=traced_model,                        # Traced PyTorch model
    input_specs={"image": input_shape},        # Input specification
    device=device,                             # Device
    options=compile_options,
)

# %% [markdown]
# Expore more compiler options <a href=https://app.aihub.qualcomm.com/docs/hub/compile_examples.html#compiling-pytorch-to-tflite> here</a>.

# %% [markdown]
# ## On-Device Performance Profiling

# %%
# Notebook cell 20
from qai_hub_models.utils.printing import print_profile_metrics_from_job

# Choose device
device = qai_hub.Device(selected_device)

# Runs a performance profile on-device
profile_job = qai_hub.submit_profile_job(
    model=target_model,                       # Compiled model
    device=device,                            # Device
)

# Print summary
profile_data = profile_job.download_profile()
print_profile_metrics_from_job(profile_job, profile_data)

# %% [markdown]
# ## Exercise: Try different compute units

# %%
# Notebook cell 22
profile_options="--compute_unit cpu"     # Use cpu 
profile_options="--compute_unit gpu"     # Use gpu (with cpu fallback) 
profile_options="--compute_unit npu"     # Use npu (with cpu fallback) 

# Runs a performance profile on-device
profile_job_expt = qai_hub.submit_profile_job(
    model=target_model,                     # Compiled model
    device=device,                          # Device
    options=profile_options,
)

# %% [markdown]
# ## On-Device Inference

# %%
# Notebook cell 24
sample_inputs = ffnet_40s.sample_inputs()
sample_inputs

# %%
# Notebook cell 25
torch_inputs = torch.Tensor(sample_inputs['image'][0])
torch_outputs = ffnet_40s(torch_inputs)
torch_outputs

# %%
# Notebook cell 26
inference_job = qai_hub.submit_inference_job(
        model=target_model,          # Compiled model
        inputs=sample_inputs,        # Sample input
        device=device,               # Device
)

# %%
# Notebook cell 27
ondevice_outputs = inference_job.download_output_data()
ondevice_outputs['output_0']

# %%
# Notebook cell 28
from qai_hub_models.utils.printing import print_inference_metrics
print_inference_metrics(inference_job, ondevice_outputs, torch_outputs)

# %% [markdown]
# ## Get ready for deployment!

# %%
# Notebook cell 30
target_model = compile_job.get_target_model()
_ = target_model.download("FFNet_40s.tflite")

# %%
# Notebook cell 31


# %%
# Notebook cell 32


# %%
# Notebook cell 33


# %%
# Notebook cell 34


# %%
# Notebook cell 35

