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
    parser.add_argument("--model_dir", default="/data/gpfs/projects/punim0478/reza/", type=str)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument("--lora_rank", default=32, type=int)
    args = parser.parse_args()

    login(os.getenv("HF_TOKEN"))
    base_model = "google/gemma-3-27b-it"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    with open(f"./results/{args.model}_{args.task}.json", "r") as f:
        results = json.load(f)
    
    if args.max_examples:
        results = results[:args.max_examples]
    
    reasoning_prompt="""
        You are given a multiple-choice question and a reasoning text. 
        Your task is to extract the answer choice according to the reasoning text.
        Output only the option letter.

        Question: {question}
        
        Reasoning text: {reasoning}
        
        Answer:
    """
  
    def create_prompt(sample):
        messages = [
            {"role": "user", "content": reasoning_prompt.format(question=sample["question"], reasoning=sample["explanation"][-1])}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    prompts = []
    for s in results:
        if len(s['answer']) == 2 and s['answer'][0] == s['answer'][1]:
            continue
        else:
            prompts.append(create_prompt(s))
                
    # prompts = [create_prompt(s) for s in results]


    sampling_params = SamplingParams(n=1,
                                    temperature=0,
                                    max_tokens=1024,
                                    stop_token_ids=[tokenizer.eos_token_id])
    
    llm = LLM(model=base_model,
            enable_lora=False,
            download_dir=args.model_dir,
            tensor_parallel_size=1)
            # gpu_memory_utilization=0.2)
            # max_model_len=8192)
    
    outputs = llm.generate(prompts,
                            sampling_params,
                            lora_request=None)
    count=0
    judged = []
    for e, r in enumerate(results):
        if len(s['answer']) == 2 and s['answer'][0] == s['answer'][1]:
            continue
            
        answer = outputs[count].outputs[0].text.strip()
        try:
            r["answer"].append(answer)
        except:
            r["answer"]= []
            r["answer"].append(answer)
        count+=1

    os.makedirs("./results/judged/", exist_ok=True)
    with open(f"./results/judged/{args.model}_{args.task}.json", "w") as f:
        json.dump(results, f)
