from typing import List
import warnings
import random
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from transformers import T5Tokenizer, T5Model


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


@dataclass
class EntryInfo:
    id: str
    post: str
    title: str


@dataclass
class EntryExtra:
    confidence: int = None


@dataclass
class SummaryInfo:
    text: str
    policy: str
    note: str


@dataclass
class Entry:
    info: EntryInfo
    summaries: List[SummaryInfo]
    split: str
    choice: int
    worker: str
    batch: str
    batch: str
    extra: EntryExtra


def parse_entry(x, data_class):
    members = {}
    for f in dataclasses.fields(data_class):
        if not f.name in x:
            if f.name == "post":
                if "article" in x:
                    members[f.name] = x["article"]
                else:
                    print("wran:", f.name)
            else:
                continue

        elif hasattr(f.type, "_name") and f.type._name == "List":
            element_type = f.type.__args__[0]
            source_list = x[f.name]
            if dataclasses.is_dataclass(element_type):
                members[f.name] = [parse_entry(a, element_type) for a in source_list]
            else:
                members[f.name] = source_list
        elif dataclasses.is_dataclass(f.type):
            members[f.name] = parse_entry(x[f.name], f.type)
        else:
            members[f.name] = x[f.name]
    return data_class(**members)


def load_dataset(data_dir: Path, val_frac = 1 / 20):
    json_files = data_dir.glob("batch*.json")

    pairs_by_id = {}

    for fn in json_files:
        print("reading ", fn)
        with fn.open(encoding="utf-8") as f:
            items: List[Entry] = list(parse_entry(json.loads(ln), Entry) for ln in f)

        for item in items:
            id = item.info.id
            if id in pairs_by_id:
                pairs_by_id[id].append(item)
            else:
                pairs_by_id[id] = [item]

    ids = list(pairs_by_id.keys())
    assert len(ids) > 100

    # split into train & val set
    val_count = max(10, int(val_frac * len(ids)))
    random.shuffle(ids)
    validation_ids = ids[:val_count]
    train_ids = ids[val_count:]

    print("training set:", len(train_ids))
    print("validation set:", len(validation_ids))

    # num_examples = [len(pairs_by_id[id]) for id in ids]
    # print(f'len: {len(ids)}; min_pairs: {min(num_examples)}; max_pairs: {max(num_examples)}; mean_pairs: { sum(num_examples) / len(num_examples)}')

    return pairs_by_id, train_ids, validation_ids


@dataclass
class TrainingEntry:
    id: str
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    summary_ids: torch.Tensor  # outputs, first half are the prefered summaries
    summary_mask: torch.Tensor


def tokenize_entry(xs: List[Entry], tokenizer: T5Tokenizer, max_text_length=512, max_summary_length=256):
    assert len(xs) > 0

    good_summaries = []
    bad_summaries = []

    id = xs[0].info.id
    text = xs[0].info.post

    if len(xs) > 64:
        xs = xs[:64]

    for i,x in enumerate(xs):
        assert x.choice >= 0 and x.choice < 2
        if x.info.post != text:
            warnings.warn(f"text mismatch {id}[{i}]")

        good_summary = x.summaries[x.choice].text
        bad_summary = x.summaries[1 - x.choice].text

        good_summaries.append(good_summary)
        bad_summaries.append(bad_summary)

    total_summaries = list(good_summaries)
    total_summaries.extend(bad_summaries)

    text_encoding = tokenizer.encode_plus(
        "Summarize: " + text,
        return_tensors="pt",
        max_length=max_text_length,
        truncation=True,
    )

    summary_encoding = tokenizer.batch_encode_plus(
        total_summaries,
        return_tensors="pt",
        padding=True,
        max_length=max_summary_length,
        truncation=True,
    )

    return TrainingEntry(
        id=id,
        input_ids=text_encoding.input_ids.repeat(len(total_summaries), 1),
        input_mask=text_encoding.attention_mask.repeat(len(total_summaries), 1),
        summary_ids=summary_encoding.input_ids,
        summary_mask=summary_encoding.attention_mask,
    )


def main():
    cache_dir = "../../hf_model_cache"
    data_dir = Path("/data/laion/openai_summarize/comparisons/")
    tokenized_data_fn = 'tokenizer_cache.pth'

    # try to load cached tokenizer results
    if Path(tokenized_data_fn).exists():
        print(f'found existing tokenized data file: {tokenized_data_fn}')
        data = torch.load(tokenized_data_fn)
        tokenized_items = data['tokenized_items']
        train_ids = data['train_ids']
        validation_ids = data['validation_ids']
        print(f'loaded items: {len(tokenized_items)}; train_ids: {len(train_ids)}; validation_ids: {len(validation_ids)};')
    else:
        pairs_by_id, train_ids, validation_ids = load_dataset(data_dir)
        print('tokenizing')
        tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir, model_max_length=512)
        tokenized_items = {id: tokenize_entry(xs, tokenizer=tokenizer) for id, xs in pairs_by_id.items()}
        data = { 'tokenized_items': tokenized_items, 'train_ids': train_ids, 'validation_ids': validation_ids }
        torch.save(data, tokenized_data_fn)
        print(f'saved {len(tokenized_items)} (train_ids: {len(train_ids)}; validation_ids: {len(validation_ids)}) to: {tokenized_data_fn}')
       
    

    return
    

    device = torch.device("cuda", 0)

    tokenize_entry(pairs_by_id[train_ids[0]], tokenizer)

    return
    model = T5Model.from_pretrained("t5-small", cache_dir=cache_dir)

    # model.to(device)

    input_texts = ["Studies have been shown that owning a dog is good for you", "Hallo, wie geht's?"]

    input_encoding = tokenizer.batch_encode_plus(
        input_texts,
        return_tensors="pt",
        padding=True,
        # max_length=50,
        # pad_to_max_length=True,
    )
    input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

    output_texts = ["Studies show that</s>", "yip, this is the case, very sure...</s>"]
    output_encoding = tokenizer.batch_encode_plus(
        output_texts, return_tensors="pt", padding=True
    )  # max_length=50, pad_to_max_length=True)

    decoder_input_ids, decoder_attention_mask = output_encoding.input_ids, output_encoding.attention_mask
    decoder_input_ids = model._shift_right(decoder_input_ids)

    # input_ids = input_ids.to(device)
    # decoder_input_ids = decoder_input_ids.to(device)

    # forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    print(dir(outputs))
    last_hidden_states = outputs.last_hidden_state

    # get indices of last elements
    last_token_indices = first_true_indices(decoder_attention_mask == 0) - 1
    last_token_indices = last_token_indices.clamp_min(0)
    print("last_token_indices", last_token_indices)

    print("last_hidden_states:", last_hidden_states.shape, last_hidden_states.shape)
    print(attention_mask)
    print(decoder_attention_mask)

    gather_index = last_token_indices.view(last_token_indices.shape[0], 1, 1).repeat(1, 1, 512)
    last_token_hidden_states = torch.gather(last_hidden_states, dim=1, index=gather_index)
    print("last_token_hidden_states", last_token_hidden_states.shape)
    print("extr0", last_token_hidden_states[0, 0, 0:10])
    print("extr1", last_token_hidden_states[1, 0, 0:10])
    print("last_hidden_states0", last_hidden_states[0, 3, 0:10])
    print("last_hidden_states1", last_hidden_states[1, 13, 0:10])


if __name__ == "__main__":
    main()
