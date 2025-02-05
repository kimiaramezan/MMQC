import argparse
import train
import data
import pickle
from tqdm import tqdm
def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    # parser.add_argument('--ques_file', type=str)
    parser.add_argument('--doc_file', type=str)
    parser.add_argument('--img_embed_dict', type=str)
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()
    model = train.MODEL_MAP[args.model]().cuda()
    # questions, img_pairs = data.read_quesfile(args.ques_file)
    docs = data.read_docfile(args.doc_file)
    img_embed_dict = data.read_img_embedding(args.img_embed_dict)
    run = data.read_run_dict(args.run)
    qrels = data.read_qrels_dict(args.qrels)
    facet_question = pickle.load(open('/home/yyuan/MQC/mqc/data/test/facet_ques.pkl','rb'))
    all_dict = pickle.load(open('/home/yyuan/MQC/mqc/data/test/all_dict.pkl','rb'))
    result_dict = {}
    if args.model_weights is not None:
        model.load(args.model_weights.name)
    if args.model == 'vl_bert':
        use_img = True
    else:
        use_img = False
    for r in tqdm(run):
        sub_run = {r:run[r]}
        for k,q in enumerate(facet_question[r]['ques']):
            sub_questions = {r:q}
            dataset = (sub_questions, docs)
            img_pairs = {r:facet_question[r]['img'][k]}
            result,_ = train.validate(model, dataset, img_pairs, img_embed_dict, sub_run, qrels,  use_img=use_img)
            result_dict[all_dict[q]] = result

    with open(args.out_path,'wb') as f:
        pickle.dump(result_dict,f,protocol=1)

if __name__ == '__main__':
    main_cli()
