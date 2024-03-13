from torch import Tensor
import torch
from typing import Type
from torch import nn
from torch.nn import Transformer
import math
from dataset import TextDataset

    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_idx_src, pad_idx_tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx_src).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx_tgt).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
    
class TransformerModel(nn.Module):
    def __init__(self,
                 dataset,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 embed_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):

        super(TransformerModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size_de = dataset.vocab_size_de
        self.vocab_size_en = dataset.vocab_size_en
        self.max_length = dataset.max_length


        self.transformer = Transformer(d_model=embed_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(embed_size, self.vocab_size_en)
        self.src_tok_emb = TokenEmbedding(self.vocab_size_de, embed_size)
        self.tgt_tok_emb = TokenEmbedding(self.vocab_size_en, embed_size)
        self.positional_encoding = PositionalEncoding(
            embed_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
    @torch.inference_mode()
    def inference(self, sentence, temp: float = 1.) -> str:
        
        self.eval()

        device = next(self.parameters()).device

        src = torch.tensor(([self.dataset.bos_id_de] +
                                   self.dataset.text2ids(sentence, 'de') + [self.dataset.eos_id_de])).unsqueeze(0).long().to(device).T
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(self.dataset.bos_id_de).type(torch.long).to(device)
        
        for i in range(self.max_length - 1):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            prob /= temp
            
            from torch.distributions.categorical import Categorical
            
            next_word = Categorical(logits=prob).sample().item()
            
#             while next_word == self.dataset.unk_id_en or next_word == self.dataset.pad_id_en or next_word == self.dataset.bos_id_en:
#                 prob[next_word] = 0
#                 next_word = Categorical(logits=prob).sample().item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.dataset.eos_id_en:
                break
        
        return self.dataset.ids2text(ys.squeeze(), 'en')