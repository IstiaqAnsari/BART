from transformers import BartForConditionalGeneration, BartConfig
import torch
from torch.nn import CrossEntropyLoss

def Get_Bart_Model(padding_idx,config = None, device='cpu',config_path="bart.config", vocab_size = None):
    if(config is None):
        config = BartConfig.from_pretrained(config_path)
    if(vocab_size is not None):
        config.vocab_size = vocab_size
    config.pad_token_id = padding_idx
    config.decoder_start_token_id = 1
    loss_fct = CrossEntropyLoss(ignore_index = config.pad_token_id)
    model = BartForConditionalGeneration(config, loss_fct = loss_fct)
    model.to(device)
    return model