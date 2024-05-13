import csv
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

class LSTM(pl.LightningModule):

    def __init__(self, hidden_dim1, hidden_dim2, output_dim, vocab_size, embedding_dim):
        super(LSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to("cuda")

        # Encoder LSTM layer
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True).to("cuda")

        # Decoder LSTM layer
        self.decoder_lstm = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True).to("cuda")
        self.fc = nn.Linear(hidden_dim2, output_dim)

        self.prot_letter_to_num = {'C': 0, 'D': 1, 'S': 2, 'Q': 3, 'K': 4, 'I': 5,
                       'P': 6, 'T': 7, 'F': 8, 'A': 9, 'G': 10, 'H': 11,
                       'E': 12, 'L': 13, 'R': 14, 'W': 15, 'V': 16, 
                       'N': 17, 'Y': 18, 'M': 18} 
        self.prot_num_to_letter = {v:k for k, v in self.prot_letter_to_num.items()}

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def tokenize(self, prot_seq):
        prot_seq = torch.tensor([self.prot_letter_to_num[x] for x in prot_seq]).to("cuda")
        return prot_seq

    def step(self, batch):
        v_region, cdr3 = batch
        v_region = v_region[0]
        cdr3 = cdr3[0]
        tokenized_seq = self.tokenize(v_region+cdr3)
        context_len = len(v_region)
        gen_len = len(cdr3)

         # train with teacher forcing
        cdr3_logits = torch.zeros((gen_len,20)).to("cuda")
        for i in range(gen_len):
            input_embedding = self.embedding(tokenized_seq[:i+context_len])
            input_encoding, _ = self.encoder_lstm(input_embedding)
            decoded_rep, _ = self.decoder_lstm(input_encoding)
            pred_logit_lasttok = self.fc(decoded_rep)[-1]
            cdr3_logits[i] = pred_logit_lasttok

        loss = self.loss_fn(cdr3_logits, tokenized_seq[context_len:])
        return loss

    def training_step(self, batch):
        loss = self.step(batch)
        self.log("train_loss", loss, batch_size=1)
        return loss

    def validation_step(self, batch):
        loss = self.step(batch)
        self.log("val_loss", loss, batch_size=1)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


def load_data():
    print("Loading data...")
    
    trb_gt = pd.read_csv("data/TRB_CDR3_human_VDJdb.tsv",sep='\t')
    trb_gt = trb_gt[trb_gt["Gene"] == "TRB"]
    with open('/home/dnori/tcr_gen/data/TRBV_human_imgt.csv', newline='') as f:
        reader = csv.DictReader(f)
        v_map = {row.pop('id'): row.pop('sequence') for row in reader}

    cdr3s = trb_gt["CDR3"].tolist()
    v_ids = trb_gt["V"].tolist()
    v_regions = [v_map[v_ids[i].split("*")[0]] for i in range(len(v_ids))]

    return list(zip(v_regions, cdr3s))

if __name__ == "__main__":

    data = load_data()

    # random split 80-20
    train_set = data[:int(len(data)*0.8)]
    test_set = data[int(len(data)*0.8):]

    train_dataloader = DataLoader(train_set, batch_size=1)
    test_dataloader = DataLoader(test_set, batch_size=1)

    model = LSTM(32, 16, 20, 20, 128)

    checkpoint_callback = ModelCheckpoint(dirpath="lstm_ckpts", 
                                            save_top_k=-1, 
                                            monitor="train_loss",
                                            every_n_train_steps=500,
                                            filename="lstm-{epoch:02d}-{train_loss:.2f}")
    trainer = pl.Trainer(
        num_sanity_val_steps=1,
        devices=[5],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        max_epochs=500,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )