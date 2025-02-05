import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from tqdm import tqdm
import jsonlines
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, AdamW
from PIL import Image
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import json
from tqdm import tqdm
from itertools import zip_longest
import numpy as np
from genre.trie import Trie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run LLaVA model with customized parameters.')
parser.add_argument('--num_beams', type=int, default=5, help='Number of beams for beam search.')
parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling.')
parser.add_argument('--do_sample', action='store_true', help='Enable sampling mode.')
parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum new tokens to generate.')
parser.add_argument('--test_dataset', type=str, required=True, help='Path to the test dataset.')
parser.add_argument('--weights', type=str, required=True, help='Path to the model weights.')
args = parser.parse_args()

class TokenGenerator:
    def __init__(self, trie, input_ids):
        self.trie = trie
        self.shift_val = 0 
        self.len_input = len(input_ids)

    def update_shift_val(self, length):
        self.shift_val = length

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        inp = input_ids[self.len_input + self.shift_val:].tolist()
        allowed_next_tokens = self.trie.get([1] + inp)

        if not allowed_next_tokens:
            self.shift_val += len(inp) + 1
            allowed_next_tokens = [1] 

        return allowed_next_tokens

class MyDataset(Dataset):
    def __init__(self, jsonl_file, image_base_path, processor, test=False):
        self.image_base_path = image_base_path
        self.processor = processor
        self.data = self.load_jsonl_data(jsonl_file)
        self.test = test

    def load_jsonl_data(self, jsonl_file):
        data = []
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                data.append({
                    "image": obj["images"],
                    "conversation": obj["conversation"],
                    "documents": obj["documents"]
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_files = self.data[idx]['image']
        conversation = self.data[idx]['conversation']
        documents = self.data[idx]['documents']

        documents_text = " ".join(documents[:5])
        
        image_path = f"{self.image_base_path}/{image_file}"
        image = Image.open(image_path).convert("RGB")

        images = ""
        for image_file in image_files:
            image_path = f"{self.image_base_path}/{image_file}"
            image = Image.open(image_path).convert("RGB")
            images = image
            break

        if not self.test:
            conversation += documents_text
        inputs = self.processor(
            images=images, 
            text=conversation, 
            return_tensors="pt", 
            padding = 'max_length', 
            max_length = 256, 
            truncation= True
        )

        inputs['labels'] = inputs['input_ids'].clone()
        inputs.pop('pixel_values', None)

        return inputs

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_ground_truth(file_path):
    ground_truth_data = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            documents = ' '.join(data['documents'])
            ground_truth_data.append(documents)
    return ground_truth_data

def calculate_metrics(generated, ground_truth, k_values=[1, 3, 5]):
    metrics = {}

    def is_permutation(g, gt):
        return sorted(g.split()) == sorted(gt.split())

    def dcg(relevances):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))

    def ndcg(generated, ground_truth, k):
        relevances = [
            1 if any(is_permutation(doc, gt) for gt in ground_truth) else 0
            for doc in generated[:k]
        ]
        ideal_relevances = sorted([1] * min(len(ground_truth), k) + [0] * (k - len(ground_truth)), reverse=True)
        return dcg(relevances) / dcg(ideal_relevances) if dcg(ideal_relevances) > 0 else 0

    ground_truth_set = set(ground_truth)
    for k in k_values:
        top_k = generated[:k]
        relevant = sum(1 for doc in top_k if any(is_permutation(doc, gt) for gt in ground_truth_set))
        metrics[f'p@{k}'] = relevant / k

    for i, doc in enumerate(generated):
        if any(is_permutation(doc, gt) for gt in ground_truth_set):
            metrics['MRR'] = 1 / (i + 1)
            break
    else:
        metrics['MRR'] = 0

    for k in [1, 3, 5]:
        metrics[f'ndcg@{k}'] = ndcg(generated, ground_truth, k)

    return metrics



model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16)
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    target_modules=find_all_linear_names(model)
)

lora_model = get_peft_model(model, lora_config)
lora_model.load_state_dict(torch.load(args.weights, weights_only = True))
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


test_input_file = './New_datas_LLaVA/reordered_test_ground_truth_full_selected.jsonl'
test_input_file = './New_datas_LLaVA/test_dataset_LLaVA_keywords_oneimage.jsonl'


test_dataset = MyDataset(jsonl_file=test_input_file, image_base_path="", processor=processor, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

lora_model.to('cuda')

ground_truth_data = load_ground_truth('./New_datas_LLaVA/reordered_test_ground_truth_full_selected.jsonl')


metric_totals = {'p@1': 0, 'p@3': 0, 'p@5': 0, 'MRR': 0, 'ndcg@1': 0, 'ndcg@3': 0, 'ndcg@5': 0}
num_samples = 0

keywords_list = []
with open('./New_datas_LLaVA/filtered_test_keywords_full.run', 'r') as run_file:
    for line in run_file:
        parts = line.split()
        if len(parts) >= 4:
            keywords = ' '.join(parts[2:5]) 
            keywords_list.append(keywords)


loop = tqdm(test_dataloader, leave=True)
    
for idx, batch in enumerate(loop):
    inputs = {key: value.squeeze(1).to(device) for key, value in batch.items()}

    ground_truth = ground_truth_data[idx]
    ground_truth_tokens = processor(text=ground_truth, return_tensors="pt").input_ids[0]
    ground_truth_tokens_split = ground_truth.split()
    grouped_ground_truth_tokens = [' '.join(ground_truth_tokens_split[i:i+3]) for i in range(0, len(ground_truth_tokens_split), 3)]

    trie = Trie()
    for doc_keywords in keywords_list[idx * 100:(idx + 1) * 100]:
        doc_tokens = processor.tokenizer.encode(doc_keywords)
        trie.add([1] + doc_tokens[1:])

    token_generator = TokenGenerator(trie, inputs['input_ids'][0])
        
    outputs = lora_model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens, 
        num_beams=args.num_beams, 
        prefix_allowed_tokens_fn=token_generator.prefix_allowed_tokens_fn, 
        temperature = args.temperature, 
        do_sample= args.do_sample
    )
    generated_texts = processor.batch_decode(outputs[:, -50:], skip_special_tokens=True)
        
    for generated_text in generated_texts:
        generated_tokens = generated_text.split()
        grouped_generated_tokens = [' '.join(generated_tokens[i:i+3]) for i in range(0, len(generated_tokens), 3)]

        print(grouped_generated_tokens)
        print(grouped_ground_truth_tokens)
        metrics = calculate_metrics(grouped_generated_tokens, grouped_ground_truth_tokens)
        print(metrics)
            
        for key in metric_totals.keys():
            metric_totals[key] += metrics[key]
        num_samples += 1


mean_metrics = {key: total / num_samples for key, total in metric_totals.items()}
print("Mean Metrics:", mean_metrics)