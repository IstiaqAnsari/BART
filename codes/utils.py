import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from bpemb import BPEmb
import pickle

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

def save_pickle(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {type(data)} at {path}")

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {type(data)} from {path}")
    return data

def SEED_EVERYTHING(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val); torch.backends.cudnn.benchmark = True


def bert_accuracy(x_ids, y_ids, y_pred):
    masked_idx = x_ids != y_ids
    masked_ids = y_ids[masked_idx]
    pred_ids = y_pred[masked_idx]
    accuracy = accuracy_score(masked_ids, pred_ids)
    return accuracy

def top_k_accuracy(x_ids, y_ids, y_pred):
    y_pred = y_pred.topk(5, dim=-1).indices
    masked_idx = x_ids != y_ids
    truth = y_ids[masked_idx]
    pred = y_pred[masked_idx, :]
    truth = truth.unsqueeze(-1).expand_as(pred)
    corr = pred.eq(truth).sum(1)
    return (sum(corr)/len(pred)).item()
    

@torch.no_grad()
def evaluate(model, valid_dataloader, device, loss_func):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_topk = AverageMeter()
    pbar = tqdm(valid_dataloader, total=len(valid_dataloader))
    model.eval() 
    for step, batch in enumerate(pbar):
        b_input = batch[0].to(device)
        b_output = batch[1].to(device)#; mask = b_input!=args.vs
        mask = batch[2].to(device)
	
        outputs = model(b_input, mask)#; print(outputs[0].shape, outputs[1].shape); print(outputs[2].shape)#, token_type_embeddings=torch.zeros_like(b_input))
        loss = loss_func(outputs.view(-1, model.vocab_size), b_output.view(-1))
        
        input_ids = batch[0]
        true_label = batch[1]
        pred_label = outputs.argmax(-1).cpu().numpy()
        
        acc_sc = bert_accuracy(input_ids, true_label, pred_label)
        acc_sc_topk = top_k_accuracy(b_input, b_output, outputs)
           
        loss_meter.update(loss.item())
        acc_meter.update(acc_sc)
        acc_meter_topk.update(acc_sc_topk)

        pbar.set_postfix({'loss':loss_meter.avg, 'accuracy': acc_meter.avg, 'topk_acc': acc_meter_topk.avg})

    return loss_meter.avg, acc_meter.avg, acc_meter_topk.avg

def get_bpe():
    bpe = BPEmb(model_file="/home/ansari/codes/spellchecker/sclstm/models/new_spm_model_v2.model",emb_file="/home/ansari/codes/spellchecker/sclstm/models/new_w2v_model_v2")
    # bpe.words[bpe.words.index("0")] = ""
    bpe.word2idx = {s:i for i,s in enumerate(bpe.words)}
    return bpe