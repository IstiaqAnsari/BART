
from logging import raiseExceptions
import torch, re, random, csv,os
import pandas as pd
import numpy as np
from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from bnlp import NLTKTokenizer
from utils import load_pickle, save_pickle
from random import randrange, sample
from sklearn.model_selection import train_test_split
from functools import partial   
from bnvocab import bn_tokenizer, normalizeText
from transformers.models.bart.modeling_bart import shift_tokens_right
import time
class Seq2SeqPreDataset(Dataset):
    def __init__(self, sents, vocab_size):
        self.src = sents
        self.vocab_size = vocab_size
        self.bpe = BPEmb(model_file="../data/bpe_models/character_seq_200.model",emb_file="../data/bpe_models/character_seq_200w2v")
        # self.tokenizer = NLTKTokenizer()
        self.tokenizer = bn_tokenizer
        self.normalizeText = normalizeText
        self.mask_token = 'ε'
        self.mask_id = self.vocab_size-1
        self.bn_char_alpha = set("অআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৠৡঁংঃ়ৎািীুূৃৄেৈোৌ্ৗৢৣ")
        self.mask_prob = 0.20
        self.pad_token = 'p'
        self.pad_token_id = self.bpe.encode_ids(self.pad_token)[0]
        self.start_token_id = 1
    def __len__(self):
        return len(self.src)

    def token_masking(self,sent,tokens):
        len_tokens = len(tokens)
        n_masks = int(self.mask_prob * len_tokens)
        masked_ids = np.random.choice(range(len_tokens),n_masks, replace=False)
        for x in masked_ids:
            tokens[x] = self.mask_token
        return " ".join(tokens),"mask"
    def token_deletion(self,sent,tokens):
        len_tokens = len(tokens)
        n_delete = int(self.mask_prob * len_tokens)
        delete_ids = sorted(np.random.choice(range(len_tokens), n_delete, replace=False),reverse=True)
        for x in delete_ids:
            del tokens[x]
        return " ".join(tokens),"delete"
    
    def span_masking(self,sent,tokens):
        len_tokens = len(tokens)
        n_masks = int(self.mask_prob * len_tokens)
        span_start_id = np.random.choice(range(len_tokens-n_masks))
        
        return " ".join(tokens[:span_start_id]+[self.mask_token]+tokens[span_start_id+n_masks:]),"span"

    def __getitem__(self, idx):
        sent = self.normalizeText(self.src[idx].strip())
        tokens = self.tokenizer(sent)
        tokens = tokens[:256]
        trg_sent = " ".join(tokens)
        sent = " ".join(tokens)
        
        objective_function = np.random.choice([self.token_masking,self.token_deletion,self.span_masking])
        src_sent,masking_type = objective_function(trg_sent, tokens)
        
        input_ids = self.bpe.encode_ids_with_bos_eos(src_sent)
        labels = self.bpe.encode_ids_with_eos(trg_sent)
        decoder_input_ids = [1] + labels[:-1]
        
        return torch.tensor(input_ids).long(),torch.tensor(decoder_input_ids).long(),torch.tensor(labels).long(), src_sent, trg_sent, masking_type
class Seq2SeqFineTuneDataset(Dataset):
    def __init__(self, sents, vocab_size, remove_start = False):
        self.src = sents
        self.vocab_size = vocab_size
        self.bpe = BPEmb(lang="bn", vs=self.vocab_size)
        # self.tokenizer = NLTKTokenizer()
        self.tokenizer = bn_tokenizer
        self.remove_start = remove_start
        self.normalizeText = normalizeText
        self.mask_token = 'ε'
        self.mask_id = self.vocab_size-1
        self.bn_char_alpha = set("অআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৠৡঁংঃ়ৎািীুূৃৄেৈোৌ্ৗৢৣ")
        self.mask_prob = 0.20
        self.pad_token = 'p'
        self.pad_token_id = self.bpe.encode_ids(self.pad_token)[0]
        self.start_token_id = 1
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        trg_sent, src_sent = self.src[idx]
        if(self.remove_start):
            trg_sent = trg_sent[7:]
            src_sent = src_sent[7:]
        input_ids = self.bpe.encode_ids(src_sent)
        input_ids = [1]+input_ids[:512]+[2]
        labels = self.bpe.encode_ids(trg_sent)
        labels = labels[:512] + [2]
        decoder_input_ids = [1] + labels[:-1]
        masking_type = "synthetic"
        return torch.tensor(input_ids).long(),torch.tensor(decoder_input_ids).long(),torch.tensor(labels).long(), src_sent, trg_sent, masking_type
def collate_fn_new(batch, padding_idx = 0):
    input_ids = pad_sequence([el[0] for el in batch],padding_value=padding_idx, batch_first=True).long()
    decoder_input_ids = pad_sequence([el[1] for el in batch],padding_value=padding_idx, batch_first=True).long()
    labels = pad_sequence([el[2] for el in batch],padding_value=padding_idx, batch_first=True).long()
    attention_mask = (input_ids!=padding_idx).long()
    decoder_attention_mask = (decoder_input_ids!=padding_idx).long()
    return input_ids, attention_mask,decoder_input_ids, decoder_attention_mask, labels
    
def get_dataloaders_pretrain(vocab_size, batch_size = 32,prototype = None):
    
    assert (vocab_size in [200]) , "Vocab size {vs} not available for BPEmb character sequence model".format(vs = vocab_size)
        
    
    # train_df = pd.read_csv('/home/sabbir_j/codes/BERT-MLM-FINAL-PHASE/data/train_final.csv',nrows = prototype)
    
    train_df = pd.read_csv('/home/sabbir_j/codes/BERT-MLM-FINAL-PHASE/data/sentence_queue_20220819_punctuation_filtered.csv',sep="\t",nrows = prototype)
    train_df = train_df[train_df['all_correct']==1]
    train_df['len'] = train_df.apply(lambda x: len(x['content']), axis = 1)
    train_df = train_df[train_df['len'] < 512]
    
    
    valid_df = pd.read_csv('/home/sabbir_j/codes/BERT-MLM-FINAL-PHASE/data/valid_final.csv',nrows = prototype)
    valid_df['len'] = valid_df.apply(lambda x: len(x['content']), axis = 1)
    valid_df = valid_df[valid_df['len'] < 512]


    train_ds = Seq2SeqPreDataset(train_df.content.values,vocab_size = vocab_size)
    valid_ds = Seq2SeqPreDataset(valid_df.content.values,vocab_size = vocab_size)

    
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, collate_fn=partial(collate_fn_new, padding_idx=train_ds.pad_token_id), 
                          num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                          shuffle=False, collate_fn=partial(collate_fn_new, padding_idx=valid_ds.pad_token_id),  
                          num_workers=0, pin_memory=True)
    return train_dl, valid_dl
def get_dataloaders_fine_tune(vocab_size, batch_size = 32,prototype = None, split = 0.05):
    
    assert (vocab_size in [1000,3000,5000,10000,25000,50000,100000, 200000]) , "Vocab size {vs} not available for BPEmb".format(vs = vocab_size)
        
    
    if(prototype is not None):
        path = "/home/ansari/codes/spellchecker/synthetic_data_for_grammar/data/pickeled_data/"
        file_id = 0
        file = f"train_data.{file_id}.pickle"
        file_path = os.path.join(path,file)
        train_data = load_pickle(file_path)
        train_data = [(a,b) for (x,y,a,b) in train_data]
        valid_data = train_data[-1000:]
        train_data = train_data[:prototype]
    else:
        train_data = []
        valid_data = []
        
        file_path = "/home/ansari/codes/spellchecker/synthetic_data_for_grammar/data/pickeled_data/AAAAALL_DATA_for_seq2seq.pickle"
        print("Reading data from ", file_path)
        st = time.perf_counter()
        data = load_pickle(file_path)
        # verb_data = load_pickle(verb_file_path)
        # data = data + verb_data
        print("Finish time ", time.perf_counter()-st, "seconds")
        train_data, valid_data = train_test_split(data, test_size = split,random_state = 1111)


    train_ds = Seq2SeqFineTuneDataset(train_data,vocab_size = vocab_size, remove_start=True)
    valid_ds = Seq2SeqFineTuneDataset(valid_data,vocab_size = vocab_size, remove_start=True)

    
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, collate_fn=partial(collate_fn_new, padding_idx=train_ds.pad_token_id), 
                          num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                          shuffle=False, collate_fn=partial(collate_fn_new, padding_idx=valid_ds.pad_token_id),  
                          num_workers=0, pin_memory=True)
    return train_dl, valid_dl

def get_dataloaders_gold_fine_tune(vocab_size, batch_size = 32,prototype = None, split = 0.05):
    
    assert (vocab_size in [1000,3000,5000,10000,25000,50000,100000, 200000]) , "Vocab size {vs} not available for BPEmb".format(vs = vocab_size)
        
    
    df = pd.read_csv("../data/annotated_sentences_v123.csv")
    df.dropna(inplace=True)
    data = [( row["corrected_content"],row["content"] ) for i,row in df.iterrows()]
    
    if(prototype):
        data = data[:prototype]
    train_data, valid_data = train_test_split(data, test_size = split,random_state = 1111)


    train_ds = Seq2SeqFineTuneDataset(train_data,vocab_size = vocab_size, remove_start=False)
    valid_ds = Seq2SeqFineTuneDataset(valid_data,vocab_size = vocab_size, remove_start=False)

    
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, collate_fn=partial(collate_fn_new, padding_idx=train_ds.pad_token_id), 
                          num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                          shuffle=False, collate_fn=partial(collate_fn_new, padding_idx=valid_ds.pad_token_id),  
                          num_workers=0, pin_memory=True)
    return train_dl, valid_dl



if __name__ == '__main__':
    sents = [
        "দেশে করোনা সংক্রমণের ঊর্ধ্বমুখী প্রবণতা অব্যাহত রয়েছে।",
        "আজ রোববার সকাল ৮টা পর্যন্ত গত ২৪ ঘণ্টায় দেশে ১ হাজার ৬৮০ জনের করোনা শনাক্ত হয়েছে।",
        "এ সময় করোনায় মৃত্যু হয়েছে দুইজনের।",
        "আগের দিন করোনা শনাক্ত হয়েছিল ১ হাজার ২৮০ জনের।"
    ]
    
    tl, vl = get_dataloaders_pretrain(vocab_size = 50000, prototype=32)
    
    tl, vl = get_dataloaders_fine_tune(vocab_size = 50000, prototype=32)
    print([x.shape for x in next(iter(tl))])
    
    exit(0)
    
    train_ds = Seq2SeqPreDataset(sents,vocab_size = 50000)
    
    for x in range(train_ds.__len__()):
        print(*train_ds.__getitem__(x),sep="\n")
        print()