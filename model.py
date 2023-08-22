import torch
import math
from torch import nn
from torchvision import transforms



class Transformer(torch.nn.Module):
    
    def __init__(self, image_features_dim, no_classes = 1000, src_max_len = 150, noHeads = 8, d_model = 768, d_ff = 3072, 
                 dropout = 0.1, noEncoder = 6, device ="cpu"):

        super(Transformer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        

        ##Input and Output Embedding##
        self.embedding = torch.nn.Linear(in_features = image_features_dim, out_features = d_model, device = device)   

        ##Positional Encoding##
        self.inp_pos_encoding = PositionalEncoding(d_model, dropout, tokens = src_max_len+1, device = device)
        
        ##Encoder##
        self.encoder = torch.nn.ModuleList([EncoderLayer(noHeads, d_model, d_ff, dropout, device = device) for i in range(noEncoder)])

        ##Final Layer with shared weights##
        self.finalLayer = torch.nn.Linear(in_features = d_model, out_features = no_classes, device = device)   
        self.softmax = torch.nn.Softmax(dim = -1)
        
        
    def encode(self, input_pos_embeddings, inputs_masks = None):
        
        
        ##Encoder forward##
        temp = input_pos_embeddings
        for layer in self.encoder:
            temp = layer(temp, inputs_masks)
        input_encodings = temp
        # print(f"input_encodings : {input_encodings.shape}")

        return input_encodings


        
    def forward(self, inputs):
        
        
        inputs_tokens = inputs.to(torch.float32).to(self.device)
        nbatches = inputs_tokens.shape[0]
        
        
        # print(f"inputs_tokens : {inputs_tokens.shape}")
        
        ##Input embeddings##
        input_embeddings = self.embedding(inputs_tokens) * math.sqrt(self.d_model)
        # print(f"input embeddings : {input_embeddings.shape}")
        
        ##Add [CLS] token##
        cls_token = self.cls_token.expand([nbatches, -1, -1])
        input_embeddings = torch.cat((cls_token, input_embeddings), dim=1)
        # print(f"input_embeddings : {input_embeddings.shape}")
        
        ##Add Positional Encoding##
        input_pos_embeddings = self.inp_pos_encoding(input_embeddings)
        # print(f"input_pos_embeddings : {input_pos_embeddings.shape}")
              
        
        ##Encoder forward##
        input_encodings = self.encode(input_pos_embeddings)
        # print(f"input_encodings : {input_encodings.shape}")

        ##Final Probabilities##
        output_proba = self.finalLayer(input_encodings)[:,0,:]
        # print(f"output_proba : {output_proba.shape}")

        return output_proba
    
    

class EncoderLayer(torch.nn.Module):
    
    
    def __init__(self, noHeads, d_model, d_ff, dropout, device ="cpu"):

        super(EncoderLayer, self).__init__()
        
        self.subLayer_1 = torch.nn.ModuleList([LayerNorm(d_model, device = device),
                                               MultiHeadAttention(noHeads = noHeads, d_model = d_model, device = device),
                                               nn.Dropout(dropout),
                                               Add(device = device)])
        
        self.subLayer_2 = torch.nn.ModuleList([LayerNorm(d_model, device = device),
                                               MLP(d_model, d_ff, device = device),
                                               nn.Dropout(dropout),
                                               Add(device = device)])
        
        

    def forward(self, input_pos_embeddings, inputs_masks = None):
        
        ##SubLayer 1 forward##
        
        normalized_inp = self.subLayer_1[0](input_pos_embeddings)
        # print(f"normalized_inp : {normalized_inp.shape}")
        normalized_att_weights = self.subLayer_1[1](key = normalized_inp,
                                         query = normalized_inp,
                                         value = normalized_inp,
                                        masks = inputs_masks)
        # print(f"normalized_att_weights : {normalized_att_weights.shape}")
        dropout_att_weights = self.subLayer_1[2](normalized_att_weights)
        # print(f"dropout_att_weights : {dropout_att_weights.shape}")
        residual_att_weights = self.subLayer_1[3](dropout_att_weights, input_pos_embeddings)
        # print(f"residual_att_weights : {residual_att_weights.shape}")

        ##SubLayer 2 forward##
        normalized_att_weights = self.subLayer_2[0](residual_att_weights)
        # print(f"normalized_att_weights : {normalized_att_weights.shape}")
        projected_att_weights = self.subLayer_2[1](normalized_att_weights)
        # print(f"projected_att_weights : {projected_att_weights.shape}")
        dropout_att_weights = self.subLayer_2[2](projected_att_weights)
        # print(f"dropout_att_weights : {dropout_att_weights.shape}")
        encodings = self.subLayer_2[3](dropout_att_weights, residual_att_weights)
        # print(f"encodings : {encodings.shape}")

        return encodings
    
    
    
class MultiHeadAttention(torch.nn.Module):
    
    
    def __init__(self, noHeads = 8, d_model = 512, device ="cpu"):

        super(MultiHeadAttention, self).__init__()

        self.d_v, self.d_k, self.noHeads = d_model // noHeads, d_model // noHeads, noHeads
        
        self.att = Attention()
        self.linearLayers = torch.nn.ModuleList([torch.nn.Linear(in_features = d_model, out_features = d_model, device = device), 
                              torch.nn.Linear(in_features = d_model, out_features = d_model, device = device), 
                              torch.nn.Linear(in_features = d_model, out_features = d_model, device = device)])
        self.finalLinear = torch.nn.Linear(in_features = noHeads*self.d_v, out_features = d_model, device = device)
        

    def forward(self, key, query, value, masks = None):

        nbatches = key.shape[0]
        
        ##Key, Query, Value projections##
        key_transf = self.linearLayers[0](key).view(nbatches, -1, self.noHeads, self.d_k).transpose(1, 2)
        query_transf = self.linearLayers[1](query).view(nbatches, -1, self.noHeads, self.d_k).transpose(1, 2)
        value_transf = self.linearLayers[2](value).view(nbatches, -1, self.noHeads, self.d_k).transpose(1, 2)

#         print(f"key_transf : {key_transf.shape}")
#         print(f"query_transf : {query_transf.shape}")
#         print(f"value_transf : {value_transf.shape}")

        if masks is not None:
            # Same mask applied to all h heads.
            masks = masks.unsqueeze(1)

        ##Projected Key, Query, Value attention weights##
        attWeight = self.att(key_transf, query_transf, value_transf, masks)
        # print(f"attWeight : {attWeight.shape}")
            
        ##Concatenate attention heads##
        concatHeads = attWeight.transpose(1, 2).contiguous().view(nbatches, -1, self.noHeads * self.d_k)
        # print(f"concatHeads : {concatHeads.shape}")

        ##Project concatenated heads##
        finalProjection = self.finalLinear(concatHeads)
        # print(f"finalProjection : {finalProjection.shape}")

        return finalProjection

        
                          
    
    
class Attention(torch.nn.Module):
    
    
    def __init__(self, device ="cpu"):

        super(Attention, self).__init__()
        
        self.softmax = torch.nn.Softmax(dim = -1)
        

    def forward(self, key, query, value, masks = None):
        
        d_k = key.shape[3]
        scores = torch.matmul(query ,torch.transpose(key,2,3))
        # print(f"scores : {scores.shape}")
        
        scaled_scores = scores/math.sqrt(d_k)
        # print(f"scaled_scores : {scaled_scores.shape}")
        
        if masks is not None:
            # print(f"masks : {masks.shape}")
            scaled_scores = scaled_scores.masked_fill(masks == 0,-1e9)
            # print(f"masked scaled_scores : {scaled_scores.shape}")
        
        normalized_scores = self.softmax(scaled_scores)
        # print(f"normalized_scores : {normalized_scores.shape}")
        
        att_weights = torch.matmul(normalized_scores , value)
        # print(f"att_weights : {att_weights.shape}")
        

        return att_weights
         
          
    
    
    
class Add(torch.nn.Module):
    
    
    def __init__(self, device ="cpu"):

        super(Add, self).__init__()
                

    def forward(self, in1, in2):
        
        return in1+in2


class LayerNorm(torch.nn.Module):
    
    
    def __init__(self, d_model, eps=1e-6, device ="cpu"):

        super(LayerNorm, self).__init__()
        
        self.a_2 = nn.Parameter(torch.ones(d_model)).to(device)
        self.b_2 = nn.Parameter(torch.zeros(d_model)).to(device)
        self.eps = eps
        

    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  
    
    
class MLP(torch.nn.Module):
    
    
    def __init__(self, d_model, d_ff, device ="cpu"):

        super(MLP, self).__init__()

        self.w1 = torch.nn.Linear(in_features = d_model, out_features = d_ff, device = device)
        self.w2 = torch.nn.Linear(in_features = d_ff, out_features = d_model, device = device)
        

    def forward(self, x):
        
        x1 = torch.nn.functional.gelu(self.w1(x))
        x2 = torch.nn.functional.gelu(self.w2(x1))
                
        return x2


class PositionalEncoding(torch.nn.Module):
    
    
    def __init__(self, d_model, dropout, tokens = 5000, device ="cpu"):

        super(PositionalEncoding, self).__init__()
        
        self.pe = torch.nn.Parameter(torch.randn(tokens, d_model))
        
        self.dropout = nn.Dropout(p=dropout)
        
        

    def forward(self, x):
        
        x = x + self.pe
        return self.dropout(x)


    
