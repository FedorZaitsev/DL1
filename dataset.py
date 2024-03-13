from variables import *
import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, save_dir: str, data_file_de: str, data_file_en: str,
                 sp_model_prefix: str = 'vocab', max_length: int = 128,
                 vocab_size_de: int = 10000, vocab_size_en: int = 10000):

        self.max_length = max_length

        
        self.PAD_ID_DE = vocab_size_de        
        if not os.path.isfile(save_dir + sp_model_prefix + '_de.model'):
            SentencePieceTrainer.train(
                pad_id=self.PAD_ID_DE,
                input=data_file_de, vocab_size=vocab_size_de + 1,
                model_type='word', model_prefix=save_dir + sp_model_prefix + '_de',
                normalization_rule_name='nmt_nfkc_cf'
            )
        
        self.sp_model_de = SentencePieceProcessor(model_file=save_dir + sp_model_prefix + '_de.model')

        with open(data_file_de) as file:
            self.texts_de = file.readlines()
        
        self.indices_de = self.sp_model_de.encode(self.texts_de)

        self.pad_id_de, self.unk_id_de, self.bos_id_de, self.eos_id_de = \
            self.sp_model_de.pad_id(), self.sp_model_de.unk_id(), \
            self.sp_model_de.bos_id(), self.sp_model_de.eos_id()
        
        self.vocab_size_de = self.sp_model_de.vocab_size()
        
        
        
        self.PAD_ID_EN = vocab_size_en
        if not os.path.isfile(save_dir + sp_model_prefix + '_en.model'):
            SentencePieceTrainer.train(
                pad_id=self.PAD_ID_EN,
                input=data_file_en, vocab_size=vocab_size_en + 1,
                model_type='word', model_prefix=save_dir + sp_model_prefix + '_en',
                normalization_rule_name='nmt_nfkc_cf'
            )       
            
        self.sp_model_en = SentencePieceProcessor(model_file=save_dir + sp_model_prefix + '_en.model')
        
        with open(data_file_en) as file:            
            self.texts_en = file.readlines()
            
        self.indices_en = self.sp_model_en.encode(self.texts_en)
        
        self.pad_id_en, self.unk_id_en, self.bos_id_en, self.eos_id_en = \
            self.sp_model_en.pad_id(), self.sp_model_en.unk_id(), \
            self.sp_model_en.bos_id(), self.sp_model_en.eos_id()
        
        self.vocab_size_en = self.sp_model_en.vocab_size()

    def text2ids(self, texts: Union[str, List[str]], lang) -> Union[List[int], List[List[int]]]:

        if lang == 'en':
            return self.sp_model_en.encode(texts)
        if lang == 'de':
            return self.sp_model_de.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]], lang) -> Union[str, List[str]]:

        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()
        if lang == 'en':
            return self.sp_model_en.decode(ids)
        if lang == 'de':
            return self.sp_model_de.decode(ids)

    def __len__(self):

        return len(self.indices_de)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
                    
        indices_de = [self.bos_id_de] + self.indices_de[item] + [self.eos_id_de]
        if len(indices_de) > self.max_length:
            indices_de = indices_de[:self.max_length]
            indices_de[-1] = self.eos_id_de
        length_de = len(indices_de)
        indices_de = torch.Tensor(indices_de)
        pad_de = torch.full(size=(self.max_length - indices_de.shape[0],), fill_value=self.pad_id_de)
        indices_de = torch.cat([indices_de, pad_de])
        
        indices_en = [self.bos_id_en] + self.indices_en[item] + [self.eos_id_en]
        if len(indices_en) > self.max_length:
            indices_en = indices_en[:self.max_length]
            indices_en[-1] = self.eos_id_en
        length_en = len(indices_en)
        indices_en = torch.Tensor(indices_en)
        pad_en = torch.full(size=(self.max_length - indices_en.shape[0],), fill_value=self.pad_id_en)
        indices_en = torch.cat([indices_en, pad_en])
        
        return indices_de, length_de, indices_en, length_en