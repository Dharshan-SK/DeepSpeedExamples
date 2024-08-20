import sys
sys.path.append("/home/ubuntu/code/applications/DeepSpeed-Chat")
import argparse
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

from dschat.utils.data.data_utils import create_prompt_dataset
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, load_state_dict_into_model
from dschat.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.perf import print_throughput
from tqdm import tqdm
from main import parse_args
from transformers import pipeline, set_seed
import time

# test data
# s3://javis-ai-parser-dev/ai_parser_testing/pharma_short_test_may_2024_20240501_20240531/outputs/20240724_2/General_Trade_IND_Purchase_Order_PHARMA_PO/

# train data
# s3://javis-ai-parser-dev/ai_parser_testing/pharma_full_prod_mar_2024_20240301_20240331/outputs/20240507_2/General_Trade_IND_Purchase_Order_PHARMA_PO/



args = parse_args()

if args.local_rank == -1:
    device = torch.device(get_accelerator().device_name())
else:
    get_accelerator().set_device(args.local_rank)
    device = torch.device(get_accelerator().device_name(), args.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()

args.global_rank = torch.distributed.get_rank()

tokenizer = load_hf_tokenizer("meta-llama/Meta-Llama-3-8B",fast_tokenizer=True)
ds_config = get_eval_ds_config(offload=args.offload, dtype="fp16", stage=args.zero_stage)
# model = create_hf_model(AutoModelForCausalLM,"meta-llama/Meta-Llama-3-8B", tokenizer, ds_config)

model = create_hf_model(AutoModelForCausalLM,"results", tokenizer, ds_config)
input_text = "Human: Please tell me about Microsoft in a few sentence? Assistant:"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
print("%"*10)
# with torch.no_grad():
    # outputs = model.generate(**inputs)
# print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
# time.sleep(1000)
# model = convert_linear_layer_to_lora(model, "layers.", 64)

model = deepspeed.init_inference(model)

print("<>"*10)
# state_dict = torch.load("results/pytorch_model.bin", map_location='cpu')
# load_state_dict_into_model(model, torch.load("results/pytorch_model.bin"), "", 2)
# load_state_dict_into_model(model.module, state_dict, "", zero_stage=2)
# model = convert_linear_layer_to_lora(model, args.lora_module_name,
#                                                      args.lora_dim)
print("><"*10)


# time.sleep(1000)
print("#"*10)
with torch.no_grad():
    outputs = model.generate(**inputs)
print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
