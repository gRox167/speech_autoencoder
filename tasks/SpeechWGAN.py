from email import generator
import torch
import soundfile as sf
import os
from pytorch_lightning import LightningModule



class SpeechWGAN(LightningModule):
    def __init__(self, generator, discriminator, loss_fn, optimizer, n_critics = 5,lr=[0.0000001, 0.0001], sampling_rate=16000, **kwargs):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.lr = lr
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.save_hyperparameters("lr")
        self.optimizer = optimizer
        self.n_critics = n_critics

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        tensorboard_logger = self.logger.experiment
        # x = batch['input_values']
        # mask = batch['attention_mask']
        length = batch['input_length']
        batch_size = batch['batch_size']
        real_samples = batch['input_values']
        ##############################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################################
        if optimizer_idx == 0:
            # when updating D net, generator need not to be change, so we can stop the gradient update
            self.generator.requires_grad_(False)
            self.discriminator.requires_grad_(True)
            # Set discriminator gradients to zero.
            self.discriminator.zero_grad()

            # Generate samples batch with G
            fake_samples = self.generator(real_samples).squeeze(1)
            #length of real_samples and fake_samples is different, so cut-off some of the real samples
            real_samples = real_samples[:,0:fake_samples.size(1)]

            # Train with real and fake
            real_samples.requires_grad_(True)
            real_outputs, real_feature_maps = self.discriminator(real_samples) # input is BL
            fake_outputs, fake_feature_maps = self.discriminator(fake_samples) # input is BL

            loss, real_losses, fake_losses, gradient_penalties = self.discriminator.discriminator_loss(real_samples, fake_samples, real_outputs, fake_outputs)
            values = {"errD": loss}  # add more items if needed
            self.log_dict(values)
            output = {
                'loss': loss,
                'progress_bar': values,
                'log': values
                }

        ##############################################
        # (2) Update G network: maximize log(D(G(z)))
        ##############################################
        elif optimizer_idx == 1:
            # when updating D net, generator need not to be change, so we can stop the gradient update
            self.generator.requires_grad_(True)
            self.discriminator.requires_grad_(False)
            # Set generator gradients to zero
            self.generator.zero_grad()

            # Generate samples batch with G
            fake_samples = self.generator(real_samples).squeeze(1)
            #length of real_samples and fake_samples is different, so cut-off some of the real samples
            real_samples = real_samples[:,0:fake_samples.size(1)]

            real_outputs, real_feature_maps = self.discriminator(real_samples) # input is BL
            fake_outputs, fake_feature_maps = self.discriminator(fake_samples)

            errG, generator_losses = self.discriminator.generator_loss(fake_outputs)
            errF = self.discriminator.feature_loss(real_feature_maps,fake_feature_maps)
            errM = self.discriminator.mse_mel_loss(fake_samples, real_samples)
            loss = errG + errF + 50*errM
            values = {"errG": errG, "errF": errF, "errM": errM}  # add more items if needed
            output = {
                'loss': loss,
                'progress_bar': values,
                'log': values
                }
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print(batch['input_length'])
        tensorboard_logger = self.logger.experiment

        length = batch['input_length']
        batch_size = batch['batch_size']
        real_samples = batch['input_values']
        fake_samples = self.generator(real_samples).squeeze(1)
        # print(batch["input_values"].shape)
        in_tensor = real_samples[:, :fake_samples.shape[-1]]

        loss = self.loss_fn(fake_samples, in_tensor)

        self.log("val_loss", loss)
        # tensorboard_logger.add_audio(
            # "val_sample", real_samples[0, :].unsqueeze(0), self.global_step)
        # self.logger.experiment.add_audio("val_sample",outputs[1,:].unsqueeze(0),self.global_step)
        # self.global_step

        # save the input file and the output
        output_num = 5 if fake_samples.shape[0] >= 5 else fake_samples.shape[0]
        for i in range(0, output_num):
            sf.write(os.path.join(self.logger.log_dir, 'val_epoch{}_input_{}.wav'.format(
                self.current_epoch, i)), in_tensor[i, :].cpu(), self.sampling_rate)
            sf.write(
                os.path.join(
                    self.logger.log_dir,
                    'val_epoch{}_output_{}.wav'.format(self.current_epoch, i)),
                fake_samples[i, :].cpu(),
                self.sampling_rate)
        return {"loss": loss}


    def configure_optimizers(self):
        lr = self.lr
        # b1 = self.b1
        # b2 = self.b2
        discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr[0])
        generator = torch.optim.Adam(self.generator.parameters(), lr=lr[1])
        return (
            {'optimizer': discriminator, 'frequency': self.n_critics},
            {'optimizer': generator, 'frequency': 1}
        )
