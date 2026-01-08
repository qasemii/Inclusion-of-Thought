import argparse
import json
import os
from huggingface_hub import login

import transformers
from utils import evaluate
from utils import get_alphabet_choice
from utils import get_yes_no
from math_utils import parse_math_boxed
from math_utils import parse_number
from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=0, type=int)
    parser.add_argument("--task", default="SQA", type=str)
    parser.add_argument("--model", default="mistral-7b", type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--model_dir", default="/data/gpfs/projects/punim0478/reza/", type=str)
    parser.add_argument("--lora_rank", default=32, type=int)
    args = parser.parse_args()
    
    if args.model == 'olmo-2-7b':
        base_model = 'allenai/OLMo-2-1124-7B-Instruct'
    elif args.model == 'olmo-2-13b':
        base_model = 'allenai/OLMo-2-1124-13B-Instruct'
    elif args.model == "llama-3.3-8b":
        base_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    else:
        raise ValueError(f"Unsupported model: {args.model}")
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    with open(f"{args.data_dir}", "r") as f: 
        test_samples = json.load(f)


    # PLAIN PROMPTING 
    reasoning_prompt = """You are given a multiple-choice question. You should reason in a step-by-step manner to get the right answer.\n\nQuestion: {question}"""
    
    def create_prompt(sample):
        messages = [
            {"role": "user", "content": reasoning_prompt.format(question=sample['question'])}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    prompts = [create_prompt(s) for s in test_samples]
    
    sampling_params = SamplingParams(n=1,
                                    temperature=0.0,
                                    max_tokens=1024,
                                    stop_token_ids=[tokenizer.eos_token_id])
    
    llm = LLM(model=base_model,
            enable_lora=False,
            download_dir=args.model_dir,
            tensor_parallel_size=1)#,
            # gpu_memory_utilization=0.4)
            # max_model_len=8192)
    
    outputs = llm.generate(prompts,
                            sampling_params,
                            lora_request=None)

    for e, output in enumerate(outputs):
        try:
            test_samples[e]["explanation"].append(output.outputs[0].text)
        except:
            test_samples[e]["explanation"]= []
            test_samples[e]["explanation"].append(output.outputs[0].text)
    
    os.makedirs("./results/", exist_ok=True)
    save_path = f"./results/{args.model}_{args.task}.json"
    with open(save_path, "w") as f:
        json.dump(test_samples, f)
