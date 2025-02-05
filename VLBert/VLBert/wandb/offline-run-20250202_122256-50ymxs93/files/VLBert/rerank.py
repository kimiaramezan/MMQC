import argparse
import train
import data


def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--ques_file', type=str)
    parser.add_argument('--doc_file', type=str)
    parser.add_argument('--img_embed_dict', type=str)
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    parser.add_argument('--img_tag_dict',type=str)
    args = parser.parse_args()
    model = train.MODEL_MAP[args.model]().cuda()
    questions, img_pairs = data.read_quesfile(args.ques_file)
    docs = data.read_docfile(args.doc_file)
    dataset = (questions, docs)
    img_embed_dict = data.read_img_embedding(args.img_embed_dict)
    img_tag_dict = data.read_img_tags(args.img_tag_dict)
    run = data.read_run_dict(args.run)
    if args.model == 'vl_bert':
        use_img = True
    else:
        use_img = False
    if args.model_weights is not None:
        model.load(args.model_weights.name)
    rerank_run = train.run_model(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, use_img=use_img, desc='rerank')
    train.write_run(rerank_run, args.out_path.name)
    # train.run_model(model, dataset, run, args.out_path.name, desc='rerank')


if __name__ == '__main__':
    main_cli()
