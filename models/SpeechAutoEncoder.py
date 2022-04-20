import torch
import torch.nn as nn
from torch.optim import AdamW
import soundfile as sf
import os

from models.HiFiGAN_Generator import HiFiGAN
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import Data2VecAudioModel
class SpeechAutoEncoder(LightningModule):
    def __init__(self,encoder_config,pretrain_encoder_flag,decoder_config,loss_fn,optimizer,lr=[0.0000001,0.0001],sampling_rate=16000,**kwargs):        
        super().__init__()
        self.sampling_rate = sampling_rate
        self.lr = lr
        self.encoder = Data2VecAudioModel.from_pretrained('./cache/models/data2vec') if pretrain_encoder_flag == True else Data2VecAudioModel(encoder_config)
        self.decoder = HiFiGAN(decoder_config)
        self.loss_fn = loss_fn
        self.save_hyperparameters("lr")
        self.optimizer = optimizer


    def forward(self,x):
        x = self.encoder(input_values=x['input_values'],attention_mask=x['attention_mask']) # here we receive input like {'input_values': tensor, 'attention_mask': tensor}
        # output of encoder shape is like (batch,length,channel)
        x = self.decoder(x.last_hidden_state.transpose(-1,-2)) 
        # output of encoder shape is like (batch,channel,length)
        return x
    
    def training_step(self, batch, batch_idx):
        # x = batch['input_values'] 
        # mask = batch['attention_mask']
        length = batch['input_length']
        outputs = self(batch).squeeze(1)
        loss = self.loss_fn(outputs,batch["input_values"][:,:outputs.shape[-1]])
        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print(batch['input_length'])
        length = batch['input_length']
        outputs = self(batch).squeeze(1)
        # print(batch["input_values"].shape)
        in_tensor = batch["input_values"][:,:outputs.shape[-1]]
        loss = self.loss_fn(outputs,in_tensor)
        self.log("val_loss",loss)
        self.logger.experiment.add_audio("val_sample",outputs[0,:].unsqueeze(0),self.global_step)
        # self.logger.experiment.add_audio("val_sample",outputs[1,:].unsqueeze(0),self.global_step)
        # self.global_step

        # save the input file and the output
        for i in range(0,5):
            sf.write(os.path.join(self.logger.log_dir,'val_epoch{}_input_{}.wav'.format(self.current_epoch,i)),in_tensor[i,:],self.sampling_rate)
        return {"loss": loss}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": self.encoder.parameters(),
                "lr": self.lr[0]
            },
            {
                "params": self.decoder.parameters(),
                "lr": self.lr[1]
            },
        ]
        optimizer = self.optimizer(optimizer_grouped_parameters)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer] #, [scheduler]
