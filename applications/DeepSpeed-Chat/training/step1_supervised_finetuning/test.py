import sys
sys.path.append("/home/ubuntu/code/applications/DeepSpeed-Chat")
import deepspeed
from transformers import AutoModel
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, load_state_dict_into_model
import torch
from dschat.utils.ds_utils import get_train_ds_config, get_eval_ds_config

# Example model and state dict
model_name = 'meta-llama/Meta-Llama-3-8B'
model = AutoModel.from_pretrained(model_name)
ds_config = get_eval_ds_config(offload=False, dtype="fp16", stage=2)
# Load checkpoint
checkpoint_path = 'results/pytorch_model.bin'
state_dict = torch.load(checkpoint_path)

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)

# Load state dict into the partitioned model
error_msgs = load_state_dict_into_model(model_to_load=model_engine.module,
                                        state_dict=state_dict,
                                        zero_stage=3)

if len(error_msgs) > 0:
    print("Errors occurred while loading state_dict:", error_msgs)
else:
    print("Model loaded successfully")

