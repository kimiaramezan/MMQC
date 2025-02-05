
import torch
import torch.nn as nn
import numpy as np
from genre.trie import MarisaTrie
import torch.nn.functional as F
import pickle


from modeling_t5 import VLT5
class VLT5GR(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        lm_labels_pos = batch["target_ids"].to(device)
        lm_labels_neg = batch["unrelated_dict_ids"].to(device)

        output_pos = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels_pos,
            return_dict=True,
            output_hidden_states=True
        )

        output_neg = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels_neg,
            return_dict=True,
            output_hidden_states=True
        )


        loss_lm_pos = output_pos.loss  # Language modeling loss for positive pairs
        loss_lm_neg = output_neg.loss  # Language modeling loss for negative pairs

        # ✅ Extract Encoder Representations
        encoder_pos = output_pos.encoder_last_hidden_state[:, 0, :]  # First token embedding
        encoder_neg = output_neg.encoder_last_hidden_state[:, 0, :]

        # ✅ Extract Decoder Representations (last decoder layer)
        if output_pos.decoder_hidden_states is not None:
            decoder_pos = output_pos.decoder_hidden_states[-1][:, 0, :]
            decoder_neg = output_neg.decoder_hidden_states[-1][:, 0, :]
        else:
            decoder_pos = output_pos.logits[:, 0, :]
            decoder_neg = output_neg.logits[:, 0, :]

        # ✅ Compute joint representations (Encoder + Decoder)
        rep_pos = F.normalize(encoder_pos + decoder_pos, p=2, dim=-1)
        rep_neg = F.normalize(encoder_neg + decoder_neg, p=2, dim=-1)

        # ✅ Compute Cosine Similarities
        positive_sim = torch.einsum('bd,bd->b', rep_pos, rep_pos)  # Sim(x, x+)
        negative_sim = torch.einsum('bd,bd->b', rep_pos, rep_neg)  # Sim(x, x-)

        # ✅ Compute Contrastive Loss (InfoNCE)
        tau = 0.1  # Temperature parameter
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim.unsqueeze(1)], dim=1) / tau
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)  # Positive examples at index 0
        loss_contrastive = F.cross_entropy(logits, labels)  # InfoNCE loss

        # ✅ Combine LM Loss & Contrastive Loss
        alpha = 0.5  # Adjust weighting factor
        loss = alpha * loss_contrastive.mean() + (1 - alpha) * (loss_lm_pos.mean() + loss_lm_neg.mean())

        # return {
        #     'loss': loss,
        #     'loss_contrastive': loss_contrastive,
        #     'loss_lm': loss_lm_pos + loss_lm_neg
        # }

        return {'loss': loss}

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        tags = batch['tag']
        topics = pickle.load(open('../datasets/keywords_output_new.pkl', 'rb'))

        batch_topics = []
        for i in range(len(input_ids)): 
            
            topic_list = []  # Use a different variable name to store topics per batch item
            for doc_id, t in topics.items(): 
                if doc_id in batch['reference'][i]:  # Check if doc_id exists in reference
                    topic_list.append([0] + self.tokenizer.encode(t))  # Append encoded topic
            batch_topics.append(MarisaTrie(topic_list)) 

        # topics = [[0] + self.tokenizer.encode(t) for t in topics.values()]
        # trie = MarisaTrie(topics)
        # print(batch_topics)
        def prefix_allowed_tokens_fn(batch_id, sent):
            return batch_topics[batch_id].get(sent.tolist()) 


        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            # tags=tags,
            num_return_sequences=10,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
        generated_scores = output['sequences_scores']
        output = output['sequences']
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        sents = []
        scores = []
        for i in range(0, len(generated_sents), 10):
            sents.append(generated_sents[i:i + 10])
            scores.append(generated_scores[i:i+10])
        result = {}
        result['pred'] = sents
        result['scores'] = scores
        return result


from modeling_bart import VLBart
class VLBartGR(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        loss = loss.mean()

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        topics = pickle.load(open('/home/yyuan/MQC/VL-T5/datasets/keywords_output_new.pkl','rb'))

        topics = [self.tokenizer.encode(t) for t in topics.values()]
        trie = MarisaTrie(topics[:2])
        print(type(self.tokenizer))
        output,scores = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            num_return_sequences=1,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            output_scores=True,
            **kwargs
        )
        # print(output)
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        # print(generated_sents)
        result = {}
        result['pred'] = generated_sents

        return result
