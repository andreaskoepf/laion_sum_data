from typing import Dict, List
import math
import uuid
import warnings
import random
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import T5Tokenizer, T5Model, get_linear_schedule_with_warmup

import wandb
import argparse


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


def load_dataset(data_dir: Path, val_frac=1 / 20):
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

    ids: List[str] = list(pairs_by_id.keys())
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

    def to(self, device: torch.DeviceObjType):
        return TrainingEntry(
            id,
            self.input_ids.to(device),
            self.input_mask.to(device),
            self.summary_ids.to(device),
            self.summary_mask.to(device),
        )


def tokenize_entry(
    xs: List[Entry], tokenizer: T5Tokenizer, max_text_length=512, max_summary_length=256, max_batch_size=64
):
    assert len(xs) > 0

    summaries = []  # good/bad summaries alternating (even indices are good, odd bad)

    id = xs[0].info.id
    text = xs[0].info.post

    if len(xs) > max_batch_size:
        xs = xs[:max_batch_size]

    for i, x in enumerate(xs):
        assert x.choice >= 0 and x.choice < 2
        if x.info.post != text:
            warnings.warn(f"text mismatch {id}[{i}]")

        good_summary = x.summaries[x.choice].text
        bad_summary = x.summaries[1 - x.choice].text

        summaries.append(good_summary)
        summaries.append(bad_summary)

    text_encoding = tokenizer.encode_plus(
        "Summarize: " + text,
        return_tensors="pt",
        max_length=max_text_length,
        truncation=True,
    )

    summary_encoding = tokenizer.batch_encode_plus(
        summaries,
        return_tensors="pt",
        padding=True,
        max_length=max_summary_length,
        truncation=True,
    )

    return TrainingEntry(
        id=id,
        input_ids=text_encoding.input_ids.repeat(len(summaries), 1),
        input_mask=text_encoding.attention_mask.repeat(len(summaries), 1),
        summary_ids=summary_encoding.input_ids,
        summary_mask=summary_encoding.attention_mask,
    )


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


class RewardModel(nn.Module):
    def __init__(self, cache_dir, init_scale=0.7, pool="last", dropout=0.0):
        super().__init__()
        assert pool in (
            "last",
            "mean",
        ), "pool type must be either last (hidden states of last tokens) or mean (mean pooling)"

        self.t5 = T5Model.from_pretrained("t5-small", cache_dir=cache_dir)
        d_model = self.t5.config.d_model
        self.reward_head_dropout = nn.Dropout(p=dropout)
        reward_head = nn.Linear(d_model, 1)
        init_std = init_scale / math.sqrt(d_model + 1)
        torch.nn.init.normal_(reward_head.weight, std=init_std)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head
        self.pool = pool

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):

        decoder_input_ids = self.t5._shift_right(decoder_input_ids)
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        last_hidden_states = outputs.last_hidden_state

        if self.pool == "last":
            last_token_indices = first_true_indices(decoder_attention_mask == 0) - 1
            last_token_indices = last_token_indices.clamp_min(0)
            batch_size, _, d_model = last_hidden_states.shape
            gather_index = last_token_indices.view(batch_size, 1, 1).repeat(1, 1, d_model)
            x = torch.gather(last_hidden_states, dim=1, index=gather_index).squeeze(1)
        else:
            x = mean_pooling(last_hidden_states, decoder_attention_mask)

        x = self.reward_head_dropout(x)
        reward = self.reward_head(x)
        return reward


@torch.no_grad()
def validate(
    rm: RewardModel,
    device: torch.DeviceObjType,
    tokenized_items: Dict[str, TrainingEntry],
    val_ids: List[str],
    max_batch_size=64,
):
    assert max_batch_size % 2 == 0

    loss_acc = []
    num_pairs = 0
    num_correct_examples = 0

    reward_sum = 0
    reward_count = 0

    rm.eval()
    for id in val_ids:
        b = tokenized_items[id]
        b = b.to(device)
        batch_size = b.input_ids.shape[0]

        for s in range(0, batch_size, max_batch_size):
            e = s + max_batch_size
            y = rm.forward(
                input_ids=b.input_ids[s:e],
                attention_mask=b.input_mask[s:e],
                decoder_input_ids=b.summary_ids[s:e],
                decoder_attention_mask=b.summary_mask[s:e],
            )

            reward_sum += y.sum().item()
            reward_count += y.size(0)

            # compute loss
            pos = y[::2]  # even: good summaries
            neg = y[1::2]  # odd: bad summaries

            loss = -torch.mean(torch.log(torch.sigmoid(pos - neg)))
            loss_acc.append(loss.item())

            num_pairs += pos.size(0)
            num_correct_examples += torch.count_nonzero(pos > neg).item()

    mean_loss = sum(loss_acc) / len(loss_acc)
    accuracy = num_correct_examples / num_pairs
    reward_bias = reward_sum / reward_count

    return mean_loss, accuracy, reward_bias


def parse_args() -> argparse.Namespace:
    # parse bool args correctly, see https://stackoverflow.com/a/43357954
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="device to use")
    parser.add_argument("--device_index", default=0, type=int, help="device index")
    parser.add_argument(
        "--manual_seed",
        default=426349901,
        type=int,
        help="initialization of pseudo-RNG",
    )
    parser.add_argument("--warmup", default=500, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--project", default="reward_model", type=str, help="project name for wandb")
    parser.add_argument(
        "--name",
        default="rm_" + uuid.uuid4().hex,
        type=str,
        help="wandb experiment name",
    )
    parser.add_argument("--pool", default="last", type=str, help="pool type: 'last' or 'mean'")
    parser.add_argument("--eval_interval", default=2000, type=int)
    parser.add_argument("--randomized", default=False, type=str2bool, help="Activate batch randomization")
    parser.add_argument("--batch_size", default=16, type=int, help="maximum batch size")
    parser.add_argument("--dropout", default=0, type=float, help="reward head dropout probability")
    parser.add_argument("--output_dir", default="./checkpoints/", type=str, help="")
    return parser.parse_args()


def write_checkpoint(epoch, step, model, optimizer, lr_scheduler, output_dir, name, args):
    dir_path = Path(output_dir)
    folder_path = dir_path / "{}_checkpoint_{:02d}_{:07d}".format(name, epoch, step)
    print(f"writing cehckpoint: {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    torch.save(args, folder_path / "args.pth")
    torch.save(model.state_dict(), folder_path / "model.pth")
    torch.save(optimizer.state_dict(), folder_path / "optimizer.pth")
    torch.save(lr_scheduler.state_dict(), folder_path / "lr_scheduler.pth")


def load_reward_model(chkpt_dir: str, hf_cache_dir: str, device: torch.DeviceObjType):
    chkpt_dir = Path(chkpt_dir)

    args = torch.load(chkpt_dir / "args.pth")
    model_data = torch.load(chkpt_dir / "model.pth", map_location=device)
    rm = RewardModel(cache_dir=hf_cache_dir, init_scale=0.7, pool=args.pool, dropout=args.dropout)
    rm.load_state_dict(model_data)
    rm.to(device)
    rm.eval()
    return rm


def main():
    print(f"Using pytorch version {torch.__version__}")
    args = parse_args()

    print("Effective args:", args)

    torch.manual_seed(args.manual_seed)
    device = torch.device(args.device, args.device_index)

    if args.wandb:
        wandb_mode = "online"
        wandb.login()
    else:
        wandb_mode = "disabled"
    wandb.init(project=args.project, config=vars(args), mode=wandb_mode, name=args.name)

    cache_dir = "../../hf_model_cache"
    # data_dir = Path("/data/laion/openai_summarize/comparisons/")
    data_dir = Path("../../openai_summarize_from_feedback/comparisons/")
    tokenized_data_fn = "tokenizer_cache.pth"

    tokenized_items: Dict[str, TrainingEntry]

    # try to load cached tokenizer results
    if Path(tokenized_data_fn).exists():
        print(f"found existing tokenized data file: {tokenized_data_fn}")
        data = torch.load(tokenized_data_fn)
        tokenized_items = data["tokenized_items"]
        train_ids = data["train_ids"]
        validation_ids = data["validation_ids"]
        print(
            f"loaded items: {len(tokenized_items)}; train_ids: {len(train_ids)}; validation_ids: {len(validation_ids)};"
        )
    else:
        pairs_by_id, train_ids, validation_ids = load_dataset(data_dir)
        print("tokenizing")
        tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir, model_max_length=512)
        tokenized_items = {id: tokenize_entry(xs, tokenizer=tokenizer) for id, xs in pairs_by_id.items()}
        data = {"tokenized_items": tokenized_items, "train_ids": train_ids, "validation_ids": validation_ids}
        torch.save(data, tokenized_data_fn)
        print(
            f"saved {len(tokenized_items)} (train_ids: {len(train_ids)}; validation_ids: {len(validation_ids)}) to: {tokenized_data_fn}"
        )

    """
    # determine model bias on all validation set entries
    rm = load_reward_model(
        "./checkpoints/rnd_b48_mean_do25_checkpoint_01_0006772", hf_cache_dir=cache_dir, device=device
    )
    mean_loss, accuracy, reward_bias = validate(rm, device, tokenized_items, validation_ids)
    print(f"mean_loss: {mean_loss:.4f}; accuracy: {accuracy:.2%}; reward_bias: {reward_bias};")
    """

    rm = RewardModel(cache_dir=cache_dir, init_scale=0.7, pool=args.pool, dropout=args.dropout)
    rm.to(device)

    epochs = args.epochs
    num_training_steps = len(train_ids) * epochs
    num_warmup_steps = args.warmup
    lr = args.lr
    max_batch_size = args.batch_size
    eval_interval = args.eval_interval

    optimizer = optim.Adam(rm.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    train_offset = 0
    loss_acc = []
    num_acc_examples = 0
    num_correct_examples = 0

    random_train_order = list()
    for id in train_ids:
        items = tokenized_items[id]
        for i in range(0, items.input_ids.size(0), 2):
            random_train_order.append((id, i))
    random.shuffle(random_train_order)

    assert max_batch_size % 2 == 0, "max_batch_size must be divisible by 2"
    epoch = 0
    step = 0
    while epoch < epochs:
        if step % eval_interval == 0:
            val_loss, val_acc, reward_bias = validate(rm, device, tokenized_items, validation_ids, max_batch_size)
            print(f"[val] step: {step}; loss: {val_loss:.4f}; acc: {val_acc:.2%};")
            wandb.log({"val.loss": val_loss, "val.acc": val_acc}, step=step)

        if args.randomized:
            # collect max_batch_size random examples

            batch_entries = []
            for i in range(0, max_batch_size, 2):
                if train_offset >= len(random_train_order):
                    random.shuffle(random_train_order)
                    train_offset = 0
                    print("| next epoch |")
                    epoch += 1
                    write_checkpoint(epoch, step, rm, optimizer, lr_scheduler, args.output_dir, args.name, args)

                entry_id, si = random_train_order[train_offset]
                train_offset += 1
                assert si % 2 == 0
                entry = tokenized_items[entry_id]
                batch_entries.append(
                    (
                        entry.input_ids[si : si + 2],
                        entry.input_mask[si : si + 2],
                        entry.summary_ids[si : si + 2],
                        entry.summary_mask[si : si + 2],
                    )
                )
            longest_input = max(be[0].size(1) for be in batch_entries)
            longest_summary = max(be[2].size(1) for be in batch_entries)

            # print('longest_input', longest_input)
            # print('longest_summary', longest_summary)

            input_ids = torch.zeros(max_batch_size, longest_input, dtype=torch.long)
            input_mask = torch.zeros(max_batch_size, longest_input, dtype=torch.long)
            summary_ids = torch.zeros(max_batch_size, longest_summary, dtype=torch.long)
            summary_mask = torch.zeros(max_batch_size, longest_summary, dtype=torch.long)

            for i, be in enumerate(batch_entries):
                j = i * 2
                l = be[0].size(1)
                input_ids[j : j + 2, :l] = be[0]
                input_mask[j : j + 2, :l] = be[1]
                l = be[2].size(1)
                summary_ids[j : j + 2, :l] = be[2]
                summary_mask[j : j + 2, :l] = be[3]

            # print('input_mask', input_mask)
            # print('input_ids', input_ids)

            b = TrainingEntry(
                id="rnd", input_ids=input_ids, input_mask=input_mask, summary_ids=summary_ids, summary_mask=summary_mask
            )

        else:
            if train_offset >= len(train_ids):
                train_offset = 0
                print("| next epoch |")
                epoch += 1
                write_checkpoint(epoch, step, rm, optimizer, lr_scheduler, args.output_dir, args.name, args)
            b = tokenized_items[train_ids[train_offset]]
            train_offset += 1

        b = b.to(device)

        # print('batch_size', b.input_ids.shape, b.summary_ids.shape)

        rm.train()

        batch_size = b.input_ids.shape[0]

        optimizer.zero_grad()
        assert max_batch_size % 2 == 0
        for s in range(0, batch_size, max_batch_size):
            e = s + max_batch_size
            y = rm.forward(
                input_ids=b.input_ids[s:e],
                attention_mask=b.input_mask[s:e],
                decoder_input_ids=b.summary_ids[s:e],
                decoder_attention_mask=b.summary_mask[s:e],
            )

            # compute loss
            pos = y[::2]  # even: good summaries
            neg = y[1::2]  # odd: bad summaries

            loss = -torch.log(torch.sigmoid(pos - neg))
            # equivalent:
            # loss = torch.log(1 + torch.exp(neg - pos))

            loss = loss.mean()
            loss.backward()
            loss_acc.append(loss.item())

            num_acc_examples += pos.shape[0]
            num_correct_examples += torch.count_nonzero(pos > neg).item()

        optimizer.step()
        lr_scheduler.step()

        if step % 100 == 0:
            train_mean_loss = sum(loss_acc) / len(loss_acc)
            train_accuracy = num_correct_examples / num_acc_examples
            print(
                f"[train] step: {step}; loss: {train_mean_loss:.4f}; acc: {train_accuracy:.2%}; lr: {lr_scheduler.get_last_lr()[0]:.4e}; (example rewards pos: {pos[0].item():.4f}; neg: {neg[0].item():.4f})"
            )
            loss_acc.clear()

            wandb.log(
                {
                    "train.loss": train_mean_loss,
                    "train.acc": train_accuracy,
                    "train.lr": lr_scheduler.get_last_lr()[0],
                },
                step=step,
            )

        step += 1


if __name__ == "__main__":
    main()
