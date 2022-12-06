from typing import List
from transformers import AutoTokenizer, AutoModel
from reward_model.train import load_reward_model
import pandas as pd
import torch
import torch.nn.functional as F


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


class ContrieverScoring:
    def __init__(self, device: torch.DeviceObjType, hf_cache_dir: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', cache_dir=hf_cache_dir)
        self.model = AutoModel.from_pretrained('facebook/contriever', cache_dir=hf_cache_dir)
        self.model.to(device)

    def embed(self, text_list: List[str]):
        tokens = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        tokens.to(self.device)
        outputs = self.model(**tokens)
        return mean_pooling(outputs[0], tokens['attention_mask'])

    def score_single(self, text, summary):
        c, d = self.score_multiple([text], [summary])
        return c.squeeze(), d.squeeze()
    
    def score_multiple(self, text_list, summary_list):
        a = self.embed(text_list)
        b = self.embed(summary_list)
        c = F.cosine_similarity(a, b)
        d = a.unsqueeze(-2) @ b.unsqueeze(-1)
        d = d.squeeze(-1).squeeze(1)
        return c, d


class RewardModelScoring:
    def __init__(self, chkpt_dir: str, hf_cache_dir: str, device: torch.DeviceObjType, bias: float=None):
        self.device = device
        self.rm = load_reward_model(chkpt_dir, hf_cache_dir=hf_cache_dir, device=device)
        self.tokenizer = self.rm.tokenizer
        self.bias = bias

    def score_single(self, text, summary):
        r = self.score_multiple([text], [summary])
        return r.squeeze()
    
    def score_multiple(self, text_list, summary_list):
        text_list = ["Summarize: " + s for s in text_list]  # prepend instruction
        text_tokens = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        summary_tokens = self.tokenizer(summary_list, padding=True, truncation=True, return_tensors='pt')
        text_tokens.to(self.device)
        summary_tokens.to(self.device)
        r = self.rm.forward(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            decoder_input_ids=summary_tokens.input_ids,
            decoder_attention_mask=summary_tokens.attention_mask
        )
        r = r.squeeze(1)
        if self.bias:
            r = r - self.bias
        return r


def filter():
    print('hallo')



def main():
    print('hello')


if __name__ == '__main__':
    main()

