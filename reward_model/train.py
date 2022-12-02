import torch
from transformers import T5Tokenizer, T5Model


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(
        row_len, dtype=dtype, device=bools.device
    )
    return torch.min(zero_or_index, dim=-1).values


def main():
    device = torch.device('cuda', 0)
    cache_dir = '../../hf_model_cache'

    tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir, model_max_length=512)
    model = T5Model.from_pretrained("t5-small", cache_dir=cache_dir)

    #model.to(device)

    input_texts = [
        "Studies have been shown that owning a dog is good for you",
        "Hallo, wie geht's?"
    ]

    input_encoding = tokenizer.batch_encode_plus(
        input_texts,
        return_tensors="pt",
        padding = True,
        # max_length=50, 
        # pad_to_max_length=True, 
    )  
    input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

    output_texts = ["Studies show that</s>", "yip, this is the case, very sure...</s>"]
    output_encoding = tokenizer.batch_encode_plus(output_texts, return_tensors="pt", padding=True) # max_length=50, pad_to_max_length=True)
    
    decoder_input_ids, decoder_attention_mask = output_encoding.input_ids, output_encoding.attention_mask
    decoder_input_ids = model._shift_right(decoder_input_ids)
    
    #input_ids = input_ids.to(device)
    #decoder_input_ids = decoder_input_ids.to(device)

    # forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    print(dir(outputs))
    last_hidden_states = outputs.last_hidden_state

    # get indices of last elements
    last_token_indices = first_true_indices(decoder_attention_mask == 0) - 1
    last_token_indices = last_token_indices.clamp_min(0)
    print('last_token_indices', last_token_indices)

    print('last_hidden_states:', last_hidden_states.shape, last_hidden_states.shape)
    print(attention_mask)
    print(decoder_attention_mask)

    gather_index = last_token_indices.view(last_token_indices.shape[0], 1, 1).repeat(1, 1, 512)
    last_token_hidden_states = torch.gather(last_hidden_states, dim=1, index=gather_index)
    print('last_token_hidden_states', last_token_hidden_states.shape)
    print('extr0', last_token_hidden_states[0,0,0:10])
    print('extr1', last_token_hidden_states[1,0,0:10])
    print('last_hidden_states0', last_hidden_states[0, 3, 0:10])
    print('last_hidden_states1', last_hidden_states[1, 13, 0:10])


if __name__ == '__main__':
    main()
