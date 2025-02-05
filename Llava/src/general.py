import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import jsonlines
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, AdamW

from PIL import Image
from tqdm import tqdm
from genre.trie import Trie
from peft import LoraConfig, get_peft_model, TaskType



import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run LLaVA model with customized parameters.')
parser.add_argument('--num_beams', type=int, default=5, help='Number of beams for beam search.')
parser.add_argument('--temperature', type=float, default=1, help='Temperature for sampling.')
parser.add_argument('--do_sample', type=bool, default = True, help='Enable sampling mode.')
parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum new tokens to generate.')
parser.add_argument('--train_dataset', type=str, required=True, help='Path to the train dataset.')
parser.add_argument('--test_dataset', type=str, required=True, help='Path to the test dataset.')
parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth')
parser.add_argument('--keywords_file', type=str, required=True, help='keywords file')
# parser.add_argument('--weights', type=str, required=True, help='Path to the model weights.')

parser.add_argument('--batch_size_train', type=int, default=2)
parser.add_argument('--batch_size_test', type=int, default=1)
parser.add_argument('--text_only', type=bool, default = False, help='train without images')
parser.add_argument('--max_length', type=int, default=256, help='tokenizer max length')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--model_save_dir', type=str, default= '/scratch/aabavandpour/llava_models/', help='Path to the model save directory')

parser.add_argument('--lora_rank', type=int, default=16)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.1)
parser.add_argument('--gpu_num', type=int, default=0)

args = parser.parse_args()
print(args)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16, 
)
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16)
processor.tokenizer.padding_side = "left"
processor.patch_size = model.config.vision_config.patch_size
processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy 


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
                    "images": obj["images"],
                    "conversation": obj["conversation"],
                    "documents": obj["documents"]
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_files = self.data[idx]['images']
        conversation = self.data[idx]['conversation']
        documents = self.data[idx]['documents']

        documents_text = " ".join(documents[:10])
        
        images = ""
        for image_file in image_files:
            image_path = f"{self.image_base_path}/{image_file}"
            image = Image.open(image_path).convert("RGB")
            images = image
            break
  

        if not self.test:
            conversation += documents_text

        if args.text_only:
            # print("gw")
            inputs = self.processor.tokenizer(
                text=conversation,
                return_tensors="pt", 
                padding = 'max_length', 
                max_length = 256, 
                truncation= True
            )
        else:
            inputs = self.processor.tokenizer(
                text=conversation,
                return_tensors="pt", 
                padding = 'max_length', 
                max_length = 256, 
                truncation= True
            )
            pix = torch.tensor(self.processor.image_processor(image)['pixel_values'])
            inputs['pixel_values'] = pix
        

        inputs['labels'] = inputs['input_ids'].clone()
        if args.text_only:
            inputs.pop('pixel_values', None)

        return inputs

# jsonl_file_train = "../New_datas_LLaVA/train_dataset_LLaVA_keywords_oneimage.jsonl"
# jsonl_file_test = "../New_datas_LLaVA/test_dataset_LLaVA_keywords.jsonl"
# jsonl_file_groundtruth = '../New_datas_LLaVA/test_dataset_LLaVA_keywords_oneimage.jsonl'
# keywords_file = '../New_datas_LLaVA/filtered_test_keywords_full.run'

jsonl_file_train = args.train_dataset
jsonl_file_test = args.test_dataset
jsonl_file_groundtruth = args.ground_truth
keywords_file = args.keywords_file

image_base_path = ""

train_dataset = MyDataset(jsonl_file=jsonl_file_train, image_base_path=image_base_path, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)

for batch in train_dataloader:
    print(batch['input_ids'])
    break
    

test_dataset = MyDataset(jsonl_file=jsonl_file_test, image_base_path=image_base_path, processor=processor, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False)

gd_dataset = MyDataset(jsonl_file=jsonl_file_groundtruth, image_base_path=image_base_path, processor=processor, test=True)
gd_dataloader = DataLoader(gd_dataset, batch_size=1, shuffle=False)

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
        """Check if two strings are permutations of each other."""
        return sorted(g.split()) == sorted(gt.split())

    def dcg(relevances):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))

    def ndcg(generated, ground_truth, k):
        relevances = [
            1 if any(is_permutation(doc, gt) for gt in ground_truth) else 0
            for doc in generated[:k]
        ]
        ideal_relevances = sorted([1] * min(len(ground_truth), k) + [0] * (k - len(ground_truth)), reverse=True)
        actual_dcg = dcg(relevances)
        ideal_dcg = dcg(ideal_relevances)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

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

    for k in k_values:
        metrics[f'ndcg@{k}'] = ndcg(generated, ground_truth, k)

    return metrics

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

lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    task_type=TaskType.CAUSAL_LM,
    target_modules=find_all_linear_names(model)
)

lora_model = get_peft_model(model, lora_config)
lora_model.train()

epochs = args.epochs
optimizer = AdamW(lora_model.parameters(), lr=args.lr)

def evaluate_model(model, test_dataloader, device):
    keywords_list = []
    metric_totals = {'p@1': 0, 'p@3': 0, 'p@5': 0, 'MRR': 0, 'ndcg@1': 0, 'ndcg@3': 0, 'ndcg@5': 0}
    num_samples = 0
    ground_truth_data = load_ground_truth(jsonl_file_groundtruth)
    with open(keywords_file, 'r', encoding='utf-8') as run_file:
        for line in run_file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:  
                keywords = parts[1].replace(',', '')  
                keywords_list.append(keywords)

    loop = tqdm(test_dataloader, leave=True)

    trie = Trie()
    for doc_keywords in keywords_list:
        doc_tokens = processor.tokenizer.encode(doc_keywords)
        trie.add([1] + doc_tokens[1:])
    
    for idx, batch in enumerate(loop):
        inputs = {key: value.squeeze(1).to(device) for key, value in batch.items()}

        ground_truth = ground_truth_data[idx]
        ground_truth_tokens = processor(text=ground_truth, return_tensors="pt").input_ids[0]
        ground_truth_tokens_split = ground_truth.split()
        grouped_ground_truth_tokens = [' '.join(ground_truth_tokens_split[i:i+3]) for i in range(0, len(ground_truth_tokens_split), 3)]

        

        token_generator = TokenGenerator(trie, inputs['input_ids'][0])
            
        outputs = lora_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens, 
            num_beams=args.num_beams, 
            prefix_allowed_tokens_fn=token_generator.prefix_allowed_tokens_fn, 
            # temperature = args.temperature, 
            # do_sample= args.do_sample
        )
        generated_texts = processor.batch_decode(outputs[:, -args.max_new_tokens:], skip_special_tokens=True)
            
        for generated_text in generated_texts:
            generated_tokens = generated_text.split()
            grouped_generated_tokens = [' '.join(generated_tokens[i:i+3]) for i in range(0, len(generated_tokens), 3)]
      
            # print(grouped_generated_tokens)
            # print(grouped_ground_truth_tokens)
            metrics = calculate_metrics(grouped_generated_tokens, grouped_ground_truth_tokens)
            # print(metrics)
                
            for key in metric_totals.keys():
                metric_totals[key] += metrics[key]
            num_samples += 1


    mean_metrics = {key: total / num_samples for key, total in metric_totals.items()}
    print("Mean Metrics:", mean_metrics)
    return mean_metrics

model.to(device)
best_mean_metric = -1 

for epoch in range(epochs):
    loop = tqdm(train_dataloader, leave=True)
    step_counter = 0
    total_loss = 0 
    num_batches = len(train_dataloader)
    
    
    for batch in loop:
        inputs = {key: value.squeeze(1).to(device) for key, value in batch.items() if value is not None}
        # print(inputs)
        if args.text_only:
            outputs = lora_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
        else:
            outputs = lora_model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
        step_counter += 1
        if step_counter % 3067 == 0:
            mean_metrics = evaluate_model(model, test_dataloader, device)
            print(f"Metrics: {mean_metrics}")
            current_mean = sum(mean_metrics.values()) / len(mean_metrics)
            print(f"score: {current_mean}")
            if current_mean > best_mean_metric:  
                best_mean_metric = current_mean 
                model_save_path = args.model_save_dir
                print(f"New best: saving...")
                torch.save(lora_model.state_dict(), model_save_path)
                print(f"New best model saved with mean metric: {current_mean:.4f}")

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} finished with average loss: {avg_loss:.4f}")