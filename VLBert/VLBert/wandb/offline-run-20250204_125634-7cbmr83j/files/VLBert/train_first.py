import os
import argparse
import subprocess
import random
import tempfile
from tqdm import tqdm
import torch
import modeling
import data
import pytrec_eval
from statistics import mean
from collections import defaultdict
import wandb
from datetime import datetime
from pathlib import Path
import pytrec_eval
from collections import defaultdict
from statistics import mean
import data_first
from transformers import BertTokenizer
import numpy as np


SEED = 42
LR = 0.001
BERT_LR = 2e-5
MAX_EPOCH = 30
BATCH_SIZE = 32
BATCHES_PER_EPOCH = 32
GRAD_ACC_SIZE = 2
#other possibilities: ndcg
VALIDATION_METRIC = 'P_5'
PATIENCE = 20 # how many epochs to wait for validation improvement

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker,
    # 'vl_bert': modeling.VanillaVisualBertRanker,
    'vl_bert': modeling.VanillaVLBertRanker,
}


def main(model, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels_train, valid_run, qrels_valid, use_img=True, model_out_dir=None):
    '''
        Runs the training loop, controlled by the constants above
        Args:
            model(torch.nn.model or str): One of the models in modelling.py, 
            or one of the keys of MODEL_MAP.
            dataset: A tuple containing two dictionaries, which contains the 
            text of documents and queries in both training and validation sets:
                ({"q1" : "query text 1"}, {"d1" : "doct text 1"} )
            img_pairs: The corresponding images of each clarification question. E.g.:
                {"q1: : ["img1", "img2", "img3"]} 
            train_pairs: A dictionary containing query document mappings for the training set
            (i.e, document to to generate pairs from). E.g.:
                {"q1: : ["d1", "d2", "d3"]}
            qrels_train(dict): A dicationary containing training qrels. Scores > 0 are considered
            relevant. Missing scores are considered non-relevant. e.g.:
                {"q1" : {"d1" : 2, "d2" : 0}}
            If you want to generate pairs from qrels, you can pass in same object for qrels_train and train_pairs
            valid_run: Query document mappings for validation set, in same format as train_pairs.
            qrels_valid: A dictionary  containing qrels
            model_out_dir: Location where to write the models. If None, a temporary directoy is used.
    '''
    if isinstance(model,str):
        model = MODEL_MAP[model]().cuda()
    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    epoch = 0
    top_valid_score = None
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)
    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels_train,use_img=use_img)
        print(f'train epoch={epoch} loss={loss}')
        wandb_log_dict = {}
        wandb_log_dict['Train/Loss'] = loss

        _,valid_score = validate(model, dataset, img_pairs, img_embed_dict, img_tag_dict, valid_run, qrels_valid, use_img=use_img)
        print(f'validation epoch={epoch} score={valid_score}')
        wandb_log_dict[f'Valid/valid_score'] = valid_score
        wandb.log(wandb_log_dict, step=epoch)
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights', flush=True)
            model.save(os.path.join(model_out_dir, 'weights.p'))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            break
        
    #load the final selected model for returning
    if top_valid_score_epoch != epoch:
        model.load(os.path.join(model_out_dir, 'weights.p'))
    return (model, top_valid_score_epoch)


def train_iteration(model, optimizer, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels,use_img=True):
    total = 0
    model.train()
    total_loss = 0.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data_first.iter_train_pairs(model, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels, GRAD_ACC_SIZE):
            if use_img:
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'],
                               record['img_embed'],
                               torch.tensor(record['tag']))
            else:
                # print(tokenizer.batch_decode(record['query_tok']))
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'])
            count = len(record['query_id']) // 2
            # count = len(record['query_id'])
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            # targets = torch.ones_like(scores)
            # loss = torch.nn.functional.mse_loss(scores, targets)
            
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


# def validate(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, valid_qrels, use_img=True):
#     run_scores = run_model(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run,use_img)
#     metric = VALIDATION_METRIC
#     if metric.startswith("P_"):
#         metric = "P"
#     trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, {metric,'P_1', 'P_3', 'P_5', 'ndcg_cut_1','ndcg_cut_3','ndcg_cut_5'})
#     eval_scores = trec_eval.evaluate(run_scores)
#     print(eval_scores)
#     metric_totals = {}
#     metric_counts = {}

#     # Iterate through each topic and its scores
#     for topic_scores in eval_scores.values():
#         for metric, score in topic_scores.items():
#             if metric not in metric_totals:
#                 metric_totals[metric] = 0
#                 metric_counts[metric] = 0
#             metric_totals[metric] += score
#             metric_counts[metric] += 1

#     # Calculate the average for each metric
#     metric_averages = {metric: metric_totals[metric] / metric_counts[metric] for metric in metric_totals}

#     print(metric_averages)
#     return mean([d[VALIDATION_METRIC] for d in eval_scores.values()])
#     p5 = 0.
#     for r in run_scores:
#         ground_truth = [i for i in valid_qrels[r] if valid_qrels[r][i]>0]
#         result = [(i,run_scores[r][i]) for i in run_scores[r]]
#         result = sorted(result,key=lambda i:i[1],reverse=True)
#         result = [i[0] for i in result]
#         p5+=len([i for i in result[:5] if i in ground_truth])/5.0
#     eval_scores = p5/len(run_scores)
#     return eval_scores, eval_scores
#     print(eval_scores)
#     return eval_scores, mean([d[VALIDATION_METRIC] for d in eval_scores.values()])



##ORIGINAL CEDR VALIDATE
def validate(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, valid_qrels, use_img=True):
    run_scores = run_model(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, use_img)
    
    metric = VALIDATION_METRIC
    if metric.startswith("P_"):
        metric = "P"
    
    metrics = {'P_1', 'P_3', 'P_5', 'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'recip_rank'}
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, metrics)
    eval_scores = trec_eval.evaluate(run_scores)

    # Compute average for each metric
    avg_scores = {m: np.mean([d[m] for d in eval_scores.values()]) for m in metrics}

    # print("Evaluation Scores per Query:\n", eval_scores)
    print("\nAverage Scores:")
    for metric, avg in avg_scores.items():
        print(f"{metric}: {avg:.4f}")

    return eval_scores, avg_scores[VALIDATION_METRIC]



# def validate(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, valid_qrels, use_img=True):
#     run_scores = run_model(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, use_img)
    
#     # Define the primary metric and all metrics to evaluate
#     primary_metric = 'P_5'
#     metrics = {'P_1','P_3','P_5', 'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5' , 'recip_rank'}
#     trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, metrics)
    
#     # Initialize storage for aggregated scores
#     aggregated_scores = defaultdict(list)
    
#     # Evaluate and accumulate scores for each idx
#     for idx_scores in run_scores.values():
#         eval_scores = trec_eval.evaluate(idx_scores)
        
#         # Collect scores per qid
#         for qid, topic_scores in eval_scores.items():
#             for metric, score in topic_scores.items():
#                 aggregated_scores[(qid, metric)].append(score)
    
#     # Calculate average scores for each qid and metric
#     averaged_scores = {}
#     for (qid, metric), scores in aggregated_scores.items():
#         if qid not in averaged_scores:
#             averaged_scores[qid] = {}
#         averaged_scores[qid][metric] = sum(scores) / len(scores)
    
#     # Calculate the average for the primary validation metric
#     primary_metric_avg = sum(
#         averaged_scores[qid]['P_5'] for qid in averaged_scores
#     ) / len(averaged_scores)
    
#     # Calculate the average of all metrics for all qids
#     overall_averages = {}
#     for metric in metrics:
#         total_score = 0
#         count = 0
#         for qid in averaged_scores:
#             if metric in averaged_scores[qid]:
#                 total_score += averaged_scores[qid][metric]
#                 count += 1
#         overall_averages[metric] = total_score / count if count > 0 else 0
    
#     print("Averaged Scores per QID:", averaged_scores)
#     print("Overall Averages:", overall_averages)
    
#     return overall_averages , primary_metric_avg


def run_model(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, use_img=True, desc='valid'):
    # rerank_run = defaultdict(dict)
    rerank_run = defaultdict(lambda: defaultdict(dict))
    ds_queries, ds_docs = dataset
    with torch.no_grad(), tqdm(total=sum(len(run[qid]) for qid in run), ncols=80, desc=desc, leave=False) as pbar:
        
        model.eval()
        for records in data_first.iter_valid_records(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, BATCH_SIZE):
            if use_img:
                scores = model(records['query_tok'],
                               records['query_mask'],
                               records['doc_tok'],
                               records['doc_mask'],
                               records['img_embed'],
                               torch.tensor(records['tag']))
            else:
                scores = model(records['query_tok'],
                               records['query_mask'],
                               records['doc_tok'],
                               records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
            # for idx, (qid, did, score) in enumerate(zip(records['query_id'], records['doc_id'], scores)):
            #     rerank_run[idx][qid][did] = score.item()
            pbar.update(len(records['query_id']))
            
    return rerank_run
    

def write_run(rerank_run, runf):
    '''
        Utility method to write a file to disk. Now unused
    '''
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    # parser.add_argument('--datafiles', type=argparse.FileType('rt', encoding='utf-8'), nargs='+')
    parser.add_argument('--ques_file', type=str)
    parser.add_argument('--doc_file', type=argparse.FileType('rt'))
    parser.add_argument('--img_tag_dict', type=str)
    parser.add_argument('--img_embed_dict',type=str)
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()
    project_name = 'vl_bert'
    wandb.init(project=project_name)
    current_time = datetime.now().strftime('%b%d_%H-%M')
    run_name = f'{current_time}'
    wandb.run.name = run_name
    wandb.config.update(args)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
        print("PyTorch Built With CUDA:", torch.backends.cudnn.enabled)
    model = MODEL_MAP[args.model]().cuda()
    wandb.watch(model)
    src_dir = Path(__file__).resolve().parent
    base_path = str(src_dir.parent)
    src_dir = str(src_dir)
    wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)
    # dataset = data.read_datafiles(args.datafiles)
    questions,img_pairs, tag_dict = data_first.read_quesfile(args.ques_file)
    docs = data_first.read_docfile(args.doc_file)
    dataset = (questions,docs)
    # img_pairs = data_first.read_img_dict(args.img_dict)
    qrels = data_first.read_qrels_dict(args.qrels)
    train_pairs = data_first.read_pair_dict(args.train_pairs)
    valid_run = data_first.read_run_dict(args.valid_run)
    img_embed_dict = data_first.read_img_embedding(args.img_embed_dict)
    img_tag_dict = data_first.read_img_tags(args.img_tag_dict)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    # we use the same qrels object for both training and validation sets
    if args.model=='vl_bert':
        use_img = True
    else:
        use_img = False

    img_tag_dict = tag_dict
    main(model, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels, valid_run, qrels, use_img, args.model_out_dir)


if __name__ == '__main__':
    main_cli()
