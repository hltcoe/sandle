import logging
from pprint import pprint

import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def main():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level='INFO')

    model_id = 'facebook/opt-6.7b'
    prompt = 'Hello,'

    main_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='balanced_low_0',
        torch_dtype=torch.float16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    input_token_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    output_token_ids = model.generate(
        input_token_ids.to(main_device), max_new_tokens=30, do_sample=True,
        temperature=0.7, num_return_sequences=10,
    )
    completions = [
        tokenizer.decode(output_token_ids[completion_num].tolist())
        for completion_num in range(output_token_ids.shape[0])
    ]

    pprint(completions)
    return completions


if __name__ == '__main__':
    main()
