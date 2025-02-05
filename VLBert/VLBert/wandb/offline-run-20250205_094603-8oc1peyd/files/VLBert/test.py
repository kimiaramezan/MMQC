import argparse
import train
import data
import pickle

def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--ques_file', type=str)
    parser.add_argument('--doc_file', type=argparse.FileType('rt'))
    parser.add_argument('--img_embed_dict', type=str)
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--img_tag_dict', type=str)
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()
    model = train.MODEL_MAP[args.model]().cuda()
    questions, img_pairs = data.read_quesfile(args.ques_file)
    # print(questions)
    # print(img_pairs)
    docs = data.read_docfile(args.doc_file)
    img_tag_dict = data.read_img_tags(args.img_tag_dict)
    dataset = (questions, docs)
    img_embed_dict = data.read_img_embedding(args.img_embed_dict)
    run = data.read_run_dict(args.run)
    # print(run)
    qrels = data.read_qrels_dict(args.qrels)
    if args.model_weights is not None:
        model.load(args.model_weights.name)
    if args.model == 'vl_bert':
        use_img = True
    else:
        use_img = False
    result,_ = train.validate(model, dataset, img_pairs, img_embed_dict,img_tag_dict, run, qrels,  use_img=use_img)
    with open(args.out_path,'wb') as f:
        pickle.dump(result,f,protocol=1)

if __name__ == '__main__':
    main_cli()
