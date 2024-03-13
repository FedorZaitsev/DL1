import torch
import os
import wandb
# import seaborn as sns
# import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
# from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import TransformerModel, create_mask


# sns.set_style('whitegrid')
# plt.rcParams.update({'font.size': 15})


# def plot_losses(train_losses: List[float], val_losses: List[float]):

#     clear_output()
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='train')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='val')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend()

#     plt.show()


def training_epoch(model, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices_de, lengths_de, indices_en, lengths_en in tqdm(loader, desc=tqdm_desc):
                
        src = indices_de.to(device).T
        tgt = indices_en.to(device).T
        
        tgt_input = tgt[:-1, :]
        
        pad_de, pad_en = loader.dataset.pad_id_de, loader.dataset.pad_id_en
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_de, pad_en, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :].long()
        
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(list(loader))
    return train_loss


@torch.no_grad()
def validation_epoch(model, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):

    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices_de, lengths_de, indices_en, lengths_en in tqdm(loader, desc=tqdm_desc):

        src = indices_de.to(device).T
        tgt = indices_en.to(device).T

        tgt_input = tgt[:-1, :]
        
        pad_de, pad_en = loader.dataset.pad_id_de, loader.dataset.pad_id_en
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_de, pad_en, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :].long()
        
        
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        val_loss += loss.item()

    val_loss /= len(list(loader))
    return val_loss

def train(model, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, root: str = '', save_dir: str = '', silent: bool = False):

    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id_en)
    
    if not silent:
        run = wandb.init(
            # Set the project where this run will be logged
            project="my-awesome-project",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": optimizer.param_groups[-1]['lr'],
                "epochs": num_epochs,
                "vocab_size": model.vocab_size_de
            },
        )
    
    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()
        
        
        train_losses += [train_loss]
        val_losses += [val_loss]
#         print(train_losses)
#         print(val_losses)
#         plot_losses(train_losses, val_losses)
        
#         preds = []
#         with open(root + 'data/val.de-en.de') as file:
#             texts = file.readlines()
#         for text in tqdm(texts, desc='Inference'):
#             preds.append(model.inference(text, temp=0.001))
#         refs = [texts]
#         from sacrebleu.metrics import BLEU
#         bleu = BLEU()
#         x = bleu.corpus_score(preds, refs).score
        
        if silent:
            continue

        with open(root + 'data/val.de-en.de') as file:
            texts = file.readlines()
        with open(save_dir + 'val.de-en.en-pred', 'w') as file:
            for text in tqdm(texts, desc='Inference'):
                file.writelines(model.inference(text, temp=0.01) + '\n')
        cmd = "cat " + save_dir + "val.de-en.en-pred | sacrebleu " + root + "data/val.de-en.en  --tokenize none --width 2 -b"
        x = float(os.popen(cmd).read())
        wandb.log({"train_loss": train_loss, "valid_loss": val_loss, "val bleu": x})
        if epoch % 3 == 0:
            torch.save(model.state_dict(), 'Transformer_epoch_' + str(epoch + 1))
        with open(root + 'data/test1.de-en.de') as file:
            texts = file.readlines()
        with open(save_dir + 'test1.de-en.en_epoch' + str(epoch + 1), 'w') as file:
            for text in tqdm(texts, desc='Inference'):
                file.writelines(model.inference(text, temp=0.01) + '\n')