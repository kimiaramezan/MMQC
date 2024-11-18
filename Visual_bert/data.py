import random
from tqdm import tqdm
import torch
import csv


# def read_quesfile(file):
#     queries = {}
#     img_dict = {}
#     lines = list(csv.DictReader(open(file), delimiter='\t'))
#     for line in tqdm(lines, desc='loading quesfile (by line)', leave=False):
#         c_id = line['facet_id'][1:]
#         topic, ques1, answer1, ques2, answer2, ques3, answer3, ques4, answer4 = line['topic'], line['question1'], line['answer1'], line['question2'], line['answer2'], line['question3'], line['answer3'], line['question4'], line['answer4']
#         queries[c_id] = topic+' '+ques1+' '+ answer1 + ' ' + ques2+' '+ answer2 + ' ' + ques3+' '+ answer3 + ' ' + ques4+' '+ answer4
#         img_pairs1,img_pairs2,img_pairs3, img_pairs4 = line['img_ids1'], line['img_ids2'], line['img_ids3'], line['img_ids4']
#         img_dict[c_id] = [img_pairs1,img_pairs2,img_pairs3, img_pairs4]
#     return queries,img_dict


def read_quesfile(file):
    queries = {}
    img_dict = {}
    lines = list(csv.DictReader(open(file), delimiter='\t'))
    
    for line in tqdm(lines, desc='loading quesfile (by line)', leave=False):
        c_id = line['facet_id'][1:]
        topic = line['topic']
        questions_answers = (
            line['question1'] + ' ' + line['answer1'] + ' ' +
            line['question2'] + ' ' + line['answer2'] + ' ' +
            line['question3'] + ' ' + line['answer3'] + ' ' +
            line['question4'] + ' ' + line['answer4']
        )
        entry = topic + ' ' + questions_answers

        if c_id in queries:
            queries[c_id].append(entry)
        else:
            queries[c_id] = [entry]

        img_pairs1 = line['img_ids1'].strip("[]").replace("'", "").split(', ')
        img_pairs2 = line['img_ids2'].strip("[]").replace("'", "").split(', ')
        img_pairs3 = line['img_ids3'].strip("[]").replace("'", "").split(', ') if line['img_ids3'] else []
        img_pairs4 = line['img_ids4'].strip("[]").replace("'", "").split(', ') if line['img_ids4'] else []
        combined_images = img_pairs1 + img_pairs2 + img_pairs3 + img_pairs4

        if c_id in img_dict:
            img_dict[c_id].append(combined_images)
        else:
            img_dict[c_id] = [combined_images]
    
    return queries, img_dict

def read_docfile(file):
    docs = {}
    for line in tqdm(file, desc='loading datafile (by line)', leave=False):
        d_id = line.split('\t')[1].rstrip()
        docs[d_id] = line.split('\t')[2].lstrip() if len(line.split('\t')) > 2 else ""
    return docs


def read_datafiles(files):
    queries = {}
    docs = {}
    for file in files:
        for line in tqdm(file, desc='loading datafile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) != 3:
                tqdm.write(f'skipping line: `{line.rstrip()}`')
                continue
            c_type, c_id, c_text = cols
            assert c_type in ('query', 'doc')
            if c_type == 'query':
                queries[c_id] = c_text
            if c_type == 'doc':
                docs[c_id] = c_text
    return queries, docs


def read_qrels_dict(file):
    result = {}
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def read_run_dict(file):
    result = {}
    for line in tqdm(file, desc='loading run (by line)', leave=False):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = float(score[:-4])
    return result

# def read_run_dict(file):
#     result = {}
#     for line in tqdm(file, desc='loading run (by line)', leave=False):
#         qid, _, docid, rank, score, _ = line.split()
#         score_value = float(score[:-4])
#         result.setdefault(qid, {}).setdefault(docid, []).append(score_value)
#     return result



def read_pairs_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid, docid = line.split()
        result.setdefault(qid, {})[docid] = 1
    return result

def read_pair_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid = line.split()[0]
        docid = line.split()[1]
        result.setdefault(qid, {})[docid] = 1
    return result


def read_img_dict(file):
    result = {}
    for line in tqdm(file, desc='loading imgs (by line)', leave=False):
        qid, ques, imgid1, imgid2, imgid3 = line.strip('\n').split('\t')
        result[qid] = {}
        result[qid]['ques'] = ques
        result[qid]['img'] = [imgid1,imgid2,imgid3] 
    return result

def read_img_embedding(file):
    import pickle
    img_embed_dict = pickle.load(open(file,'rb'))
    return img_embed_dict

def read_img_tags(file):
    import json
    img_tags = json.load(open(file,'r'))
    img_tags = {i['facet_id']:1 for i in img_tags}
    return img_tags


def iter_train_pairs(model, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels, batch_size, use_img=True):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'img_embed': [], 'tag': []}
    for qid, did, query_tok, doc_tok, img_embed, tag in _iter_train_pairs(model, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        batch['tag'].append(tag)
        if use_img:
            batch['img_embed'].append(img_embed)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'img_embed': [], 'tag': []}



# def _iter_train_pairs(model, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels):
#     ds_queries, ds_docs = dataset
#     while True:
#         qids = list(train_pairs.keys())
#         # random.shuffle(qids)
#         for qid in qids:
#             # pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
#             pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0)]
#             if len(pos_ids) == 0:
#                 tqdm.write("no positive labels for query %s " % qid)
#                 continue
#             pos_id = random.choice(pos_ids)
#             pos_ids_lookup = set(pos_ids)
#             pos_ids = set(pos_ids)
#             # neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_lookup]
#             # if len(neg_ids) == 0:
#             #     tqdm.write("no negative labels for query %s " % qid)
#             #     continue
#             # neg_id = random.choice(neg_ids)
#             # print(qid, pos_id)
#             query_tok = model.tokenize(ds_queries[qid])
            
#             pos_doc = ds_docs.get(pos_id)
        
#             # neg_doc = ds_docs.get(neg_id)
#             if pos_doc is None:
#                 tqdm.write(f'missing doc {pos_id}! Skipping')
#                 continue
#             # if neg_doc is None:
#             #     tqdm.write(f'missing doc {neg_id}! Skipping')
#             #     continue
#             #################
#             print(img_pairs[qid])
#             if img_pairs[qid][0] in img_embed_dict and img_pairs[qid][1] in img_embed_dict \
#                 and img_pairs[qid][2] in img_embed_dict:
#                 img_embed1 = torch.tensor(img_embed_dict[img_pairs[qid][0]]['features']).float()
#                 img_embed2 = torch.tensor(img_embed_dict[img_pairs[qid][1]]['features']).float()
#                 img_embed3 = torch.tensor(img_embed_dict[img_pairs[qid][2]]['features']).float()
#             else:
#                 tqdm.write("less than 3 images for %s " % qid)
#                 try:
#                     img_embed1 = torch.tensor(img_embed_dict[img_pairs[qid][0]]['features']).float()
#                 except:
#                     img_embed1 = torch.tensor(img_embed_dict[img_pairs[qid][1]]['features']).float()
#                 try:
#                     img_embed2 = torch.tensor(img_embed_dict[img_pairs[qid][1]]['features']).float()
#                 except:
#                     img_embed2 = img_embed1
#                 try:
#                     img_embed3 = torch.tensor(img_embed_dict[img_pairs[qid][2]]['features']).float()
#                 except:
#                     img_embed3 = img_embed2
            
            
#             img_embed = torch.stack([img_embed1, img_embed2, img_embed3]) #,
#             tag = torch.tensor(img_tag_dict[qid]).float()
#             yield qid, pos_id, query_tok, model.tokenize(pos_doc), img_embed, tag
#             # yield qid, neg_id, query_tok, model.tokenize(neg_doc), img_embed, tag

def _iter_train_pairs(model, dataset, img_pairs, img_embed_dict, img_tag_dict, train_pairs, qrels):
    ds_queries, ds_docs = dataset
    while True:
        qids = list(train_pairs.keys())
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0)]
            if len(pos_ids) == 0:
                tqdm.write(f"no positive labels for query {qid}")
                continue
            pos_id = random.choice(pos_ids)

            query_image_pairs = zip(ds_queries[qid], img_pairs[qid])

            for query, images in query_image_pairs:
                query_tok = model.tokenize(query)
                pos_doc = ds_docs.get(pos_id)
                
                if pos_doc is None:
                    tqdm.write(f'missing doc {pos_id}! Skipping')
                    continue

                if len(images) < 6:
                    tqdm.write(f"less than 6 images for {qid}")
                    continue

                num_images = 6 if len(images) == 6 else (9 if len(images) >= 9 else 12)

                try:
                    img_embeds = [
                        torch.tensor(img_embed_dict[img_id]['features']).float()
                        for img_id in images[:num_images]
                    ]
                except KeyError:
                    tqdm.write(f"missing image embeddings for {qid}")
                    continue

                # Pad the image embeddings to have 12 entries
                while len(img_embeds) < 12:
                    img_embeds.append(torch.zeros_like(img_embeds[0]))

                img_embed = torch.stack(img_embeds)
                tag = torch.tensor(img_tag_dict["F" + qid]).float()

                yield qid, pos_id, query_tok, model.tokenize(pos_doc), img_embed, tag


def iter_valid_records(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'img_embed': [], 'tag': []}
    for qid, did, query_tok, doc_tok, img_embed, tag in _iter_valid_records(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        batch['img_embed'].append(img_embed)
        batch['tag'].append(tag)
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'img_embed': [], 'tag':[]}
    # final batch
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(batch)


# def _iter_valid_records(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run):
#     ds_queries, ds_docs = dataset
#     for qid in run:
#         query_tok = model.tokenize(ds_queries[qid])
#         print(img_pairs[qid][0])
#         if img_pairs[qid][0] in img_embed_dict and img_pairs[qid][1] in img_embed_dict \
#                 and img_pairs[qid][2] in img_embed_dict:
#             img_embed1 = torch.tensor(img_embed_dict[img_pairs[qid][0]]['features']).float()
#             img_embed2 = torch.tensor(img_embed_dict[img_pairs[qid][1]]['features']).float()
#             img_embed3 = torch.tensor(img_embed_dict[img_pairs[qid][2]]['features']).float()
#         else:
#             tqdm.write("less than 3 images for %s " % qid)
#             try:
#                 img_embed1 = torch.tensor(img_embed_dict[img_pairs[qid][0]]['features']).float()
#             except:
#                 img_embed1 = torch.tensor(img_embed_dict[img_pairs[qid][1]]['features']).float()
#             try:
#                 img_embed2 = torch.tensor(img_embed_dict[img_pairs[qid][1]]['features']).float()
#             except:
#                 img_embed2 = img_embed1
#             try:
#                 img_embed3 = torch.tensor(img_embed_dict[img_pairs[qid][2]]['features']).float()
#             except:
#                 img_embed3 = img_embed2
#         img_embed = torch.stack([img_embed1, img_embed2, img_embed3]) #
#         # print(img_embed.size())
#         tag = torch.tensor(img_tag_dict[qid]).float()
#         for did in run[qid]:
#             doc = ds_docs.get(did)
#             if doc is None:
#                 tqdm.write(f'missing doc {did}! Skipping')
#                 continue
#             doc_tok = model.tokenize(doc)
            
#             yield qid, did, query_tok, doc_tok, img_embed, tag

def _iter_valid_records(model, dataset, img_pairs, img_embed_dict, img_tag_dict, run):
    ds_queries, ds_docs = dataset
    for qid in run:
        query_image_pairs = zip(ds_queries[qid], img_pairs[qid])

        for query, images in query_image_pairs:
            query_tok = model.tokenize(query)

            if len(images) < 6:
                tqdm.write(f"less than 6 images for {qid}")
                continue

            num_images = 6 if len(images) == 6 else (9 if len(images) >= 9 else 12)

            try:
                img_embeds = [
                    torch.tensor(img_embed_dict[img_id]['features']).float()
                    for img_id in images[:num_images]
                ]
            except KeyError:
                tqdm.write(f"missing image embeddings for {qid}")
                continue

            # Pad the image embeddings to have 12 entries
            while len(img_embeds) < 12:
                img_embeds.append(torch.zeros_like(img_embeds[0]))

            img_embed = torch.stack(img_embeds)
            tag = torch.tensor(img_tag_dict["F" + qid]).float()

            for did in run[qid]:
                doc = ds_docs.get(did)
                if doc is None:
                    tqdm.write(f'missing doc {did}! Skipping')
                    continue
                doc_tok = model.tokenize(doc)
                
                yield qid, did, query_tok, doc_tok, img_embed, tag


def _pack_n_ship(batch):
    QLEN = 40
    MAX_DLEN = 800
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'tag': batch['tag'],
        'query_tok': _pad_crop(batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
        'query_mask': _mask(batch['query_tok'], QLEN),
        'doc_mask': _mask(batch['doc_tok'], DLEN),
        'img_embed': torch.stack(batch['img_embed']).float().cuda()
    }


def _pad_crop(items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    return torch.tensor(result).long().cuda()


def _mask(items, l):
    result = []
    for item in items:
        # needs padding (masked)
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        # no padding (possible crop)
        if len(item) >= l:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    return torch.tensor(result).float().cuda()
