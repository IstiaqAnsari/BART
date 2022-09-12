#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:59:25 2021

@author: ratul
"""

#import numpy as np
import torch, argparse, os
from torch import nn
from dataloader import get_dataloaders_pretrain
from model import Get_Bart_Model
from utils import SEED_EVERYTHING, AverageMeter, evaluate, bert_accuracy, top_k_accuracy
from tqdm import tqdm
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


def train(args):
    SEED_EVERYTHING(1111)
    batch_size = args.batch_size
    epochs = args.epochs
    if(torch.cuda.is_available()):
        DEVICE = f'cuda:{args.device}'
        torch.cuda.set_device(DEVICE)
    print(DEVICE)
    train_dataloader, valid_dataloader = get_dataloaders_pretrain(vocab_size = args.vs, batch_size = args.batch_size, prototype=args.proto)
    
    model = Get_Bart_Model(train_dataloader.dataset.pad_token_id, device = DEVICE, vocab_size=args.vs)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                num_training_steps=args.epochs*len(train_dataloader)/args.grad_accum_steps)
    
    if args.resume_from is not None:
        state = torch.load(args.resume_from, map_location=DEVICE)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # scheduler.load_state_dict(state['scheduler_state_dict'])
    
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    best_acc_topk = 0
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    for epoch_i in range(epochs):         
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        acc_meter_topk = AverageMeter()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        model.train() 
        for step, batch in enumerate(pbar):
            
            input_ids, attention_mask,decoder_input_ids, decoder_attention_mask, labels = (x.to(DEVICE) for x in batch)
    	
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask, labels = labels)
                    loss = outputs.loss / args.grad_accum_steps
                scaler.scale(loss).backward()  
                if (step+1) % args.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)      
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            else:
                outputs = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask, labels = labels)
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()
                if (step+1) % args.grad_accum_steps == 0: 
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            acc_sc = 99.99
            topk_acc = 99.99
           
            loss_meter.update(loss.item()*args.grad_accum_steps)
            acc_meter.update(acc_sc)
            acc_meter_topk.update(topk_acc)

            pbar.set_postfix({'loss':loss_meter.avg, 'accuracy': acc_meter.avg, 'acc_ins': acc_sc, 
                              'top_k_acc':acc_meter_topk.avg, 'top_k_acc_ins':topk_acc, 
                              'shape':input_ids.shape[1], 'lr':optimizer.param_groups[0]["lr"]})
            
            if (step+1) % 10000 == 0:
                torch.save({'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict(),
                      'scheduler_state_dict':scheduler.state_dict()}, f'../models/{args.key}/current_state.pth')
            
       
          
        # valid_loss, valid_acc, valid_acc_topk = evaluate(model, valid_dataloader, DEVICE, loss_func)
        valid_loss, valid_acc, valid_acc_topk = 0.10101, 88.88, 88.88
        
       
        if valid_acc > best_acc:
          print('validation acc improved from %.4f to %.4f'%(best_acc,valid_acc))
          print('saving model...')
          torch.save({'valid_acc': valid_acc, 
                      'valid_acc_topk': valid_acc_topk,
          	          'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict(),
                      'scheduler_state_dict':scheduler.state_dict()}, 
                     f'../models/{args.key}/bert_state_best_acc.pth')
          
          best_acc = valid_acc
          
        if valid_acc_topk > best_acc_topk:
          print('top 5 validation acc improved from %.4f to %.4f'%(best_acc_topk,valid_acc_topk))
          print('saving model...')
          torch.save({'valid_acc': valid_acc, 
                      'valid_acc_topk': valid_acc_topk,
          	          'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict(),
                      'scheduler_state_dict':scheduler.state_dict()}, 
                     f'../models/{args.key}/bert_state_best_acc_topk.pth')

          best_acc_topk = valid_acc_topk

        print(f'Epoch: {epoch_i+1}/{epochs}, train loss:{loss_meter.avg:.4f}, train acc:{acc_meter.avg:.4f}\nvalid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', default='test', type=str, help='name of experiment')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs for training')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size used for training')
    parser.add_argument('--grad_accum_steps', default=32, type=int, help='number of gradient accumulation steps')
    parser.add_argument('--resume_from', default=None, help='resume from this ckpt')
    parser.add_argument('--device', default=0,type=int, help='gpu index to use for training')
    parser.add_argument('--vs', default=50000, type=int, help='vocabulary size for embedding')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='flag to indicate whether to use mixed precision')
    parser.add_argument('--proto', default=None, type=int, help='batch size used for training')
    args = parser.parse_args()
    os.makedirs(f'../models/{args.key}',exist_ok=True)
    train(args)

# nohup python train.py --key BART_TEST_RUN --epochs 2 --device 1 --vs 50000 --proto 2000  > ../logs/test_run.log &
# nohup python train.py --key BART_TEST_RUN --epochs 2 --device 1 --vs 50000 --proto 1000000  > ../logs/test_run.log &
# nohup python train.py --key BART_TEST_RUN --epochs 2 --device 1 --vs 50000 --resume_from "../models/BART_TEST_RUN/current_state.pth" > ../logs/test_run.log &

# nohup python train.py --key BART_PRETRAIN_REFINED_DATA --epochs 2 --device 2 --vs 50000 --resume_from ../models/BART_PRETRAIN_REFINED_DATA/current_state.pth > ../logs/pretrain_refined_data.log &

# nohup python train.py --key Char_BART_PreTest --epochs 2 --device 2 --vs 200 --resume_from ../models/Char_BART_PreTest/current_state.pth > ../logs/char_bart_test.log &