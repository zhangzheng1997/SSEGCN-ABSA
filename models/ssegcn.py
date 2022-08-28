'''
Description:  
Author: zhangzheng
Date: 2022-05-01 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import DefaultDict


class SSEGCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        in_dim = opt.hidden_dim
        self.opt = opt
        self.gcn_model = GCNAbsaModel(embedding_matrix=embedding_matrix, opt=opt)
        self.classifier = nn.Linear(in_dim, opt.polarities_dim)

    def forward(self, inputs):
        outputs = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, None


class GCNAbsaModel(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.embedding_matrix = embedding_matrix
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None        # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None    # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l ,short_mask= inputs            
        maxlen = max(l.data)            
        mask = mask[:, :maxlen]   

        h = self.gcn(inputs)    
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                        
        p = len(mask)
        b = len(mask[0])
        mask = mask.unsqueeze(-1).repeat(1,1,self.opt.hidden_dim)    
        outputs = (h*mask).sum(dim=1) / asp_wn                     
 
        return outputs       

class GCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):  
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = opt.embed_dim+opt.post_dim+opt.pos_dim    
        self.emb, self.pos_emb, self.post_emb = embeddings

        input_size = self.in_dim   
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers, batch_first=True, \
                dropout=opt.rnn_dropout, bidirectional=opt.bidirect)
        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2
        else:
            self.in_dim = opt.rnn_hidden


        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        self.W = nn.Linear(self.in_dim, self.in_dim)  
        self.Wxx= nn.Linear(self.in_dim, self.mem_dim)  
        self.aggregate_W = nn.Linear(self.in_dim*3, self.mem_dim)       

        self.attention_heads = opt.attention_heads
        self.head_dim = self.mem_dim // self.layers  
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim*2)  
        self.weight_list = nn.ModuleList()   
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))


        self.Wx= nn.Linear(self.attention_heads+self.mem_dim*4, self.attention_heads) 

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect) 
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True) 
        return rnn_outputs

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l,short_mask = inputs    
        src_mask = (tok != 0).unsqueeze(-2)   
        maxlen = max(l.data)   
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]  
        short_mask=short_mask[:,:,:maxlen,:maxlen]

        word_embs = self.emb(tok) 
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)     
        embs = self.in_drop(embs)

        self.rnn.flatten_parameters()
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))  
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                       
        mask = mask.unsqueeze(-1).repeat(1,1,self.opt.hidden_dim*2)     
        mask=mask[:, :maxlen, :]
        aspect_outs = (gcn_inputs*mask).sum(dim=1) / asp_wn  
        
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask,short_mask,aspect_outs)  
        weight_adj=attn_tensor  
        gcn_outputs=gcn_inputs  
        layer_list = [gcn_inputs]

        for i in range(self.layers):
            gcn_outputs = gcn_outputs.unsqueeze(1).expand(len(l), self.attention_heads, maxlen, self.mem_dim*2)   
            Ax = torch.matmul(weight_adj, gcn_outputs)  
            Ax = Ax.mean(dim=1)  
  
            Ax = self.W(Ax)   
            weights_gcn_outputs = F.selu(Ax)

            gcn_outputs = weights_gcn_outputs      
            layer_list.append(gcn_outputs)
            gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs 

            weight_adj=weight_adj.permute(0, 2, 3, 1).contiguous()      
            node_outputs1 = gcn_outputs.unsqueeze(1).expand(len(l), maxlen, maxlen, self.mem_dim*2)   
            node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous() 

            node = torch.cat([node_outputs1, node_outputs2], dim=-1) 
            edge_n=torch.cat([weight_adj,node], dim=-1)
            edge = self.Wx(edge_n) 
            edge = self.gcn_drop(edge) if i < self.layers - 1 else edge 
            weight_adj=edge.permute(0,3,1,2).contiguous() 
   
 

        # outputs = torch.cat(layer_list, dim=-1)
        node_outputs=self.Wxx(gcn_outputs)
        # node_outputs=self.aggregate_W(outputs)
        node_outputs=F.relu(node_outputs)
  

        return node_outputs  


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True): 
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)   
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, short, aspect, weight_m, bias_m, mask, dropout,):  
    d_k = query.size(-1)   

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  
    batch=len(scores)
    p=weight_m.size(0)
    max=weight_m.size(1)
    # weight_m=weight_m.unsqueeze(0).repeat(batch,1,1,1)
    weight_m=weight_m.unsqueeze(0).expand(batch,p,max,max)
   
    aspect_scores = torch.tanh(torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), bias_m)) # [16,5,41,41]
    scores=torch.add(scores,aspect_scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) 

    scores=torch.add(scores, short).cuda()
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
  
    return p_attn 


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):  
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h   
        self.h = h   
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, mask,short, aspect):  
        mask = mask[:, :, :query.size(1)]    
        if mask is not None:
            mask = mask.unsqueeze(1)   
        nbatches = query.size(0)      
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  
                             for l, x in zip(self.linears, (query, key))]      

        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)    
        aspect = self.dense(aspect)       
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k) 
        attn = attention(query, key, short, aspect, self.weight_m, self.bias_m, mask, self.dropout)   
        return attn


