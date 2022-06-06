import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils.losses import L2_MelLoss
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
        return y_d_rs, fmap_rs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
        AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs


class SpeechDiscriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mp = MultiPeriodDiscriminator()
        self.ms = MultiScaleDiscriminator()
        self.mse_mel_loss_fn = L2_MelLoss()

    def forward(self, x):
        # here we receive input like {'input_values': tensor, 'attention_mask': tensor}
        x = x.unsqueeze(1)
        mp_outputs, mp_fmaps = self.mp(x)
        ms_outputs, ms_fmaps = self.ms(x)
        # output of encoder shape is like (batch,length,channel)
        return mp_outputs+ms_outputs, mp_fmaps+ms_fmaps
    
    def discriminator_loss(self, real_samples, fake_samples, real_outputs, fake_outputs):
        loss = 0
        real_losses = []
        fake_losses = []
        gradient_penalties = []
        for real_output, fake_output in zip(real_outputs, fake_outputs):
            # Calculate W-div gradient penalty
            gradient_penalty = self.calculate_gradient_penalty(real_samples, fake_samples,
                                                                real_output, fake_output, 2, 6,
                                                                real_samples.device)
            errD_real = torch.mean(real_output)
            errD_fake = torch.mean(fake_output)
            # Add the gradients from the all-real and all-fake batches
            loss += (-errD_fake + errD_real + gradient_penalty)
            real_losses.append(errD_real.item())
            fake_losses.append(errD_fake.item())
            gradient_penalties.append(gradient_penalty.item())
        return loss, real_losses, fake_losses,gradient_penalties

    def generator_loss(self, fake_outputs):
        loss = 0
        generator_losses = []
        for fake_output in fake_outputs:
            l = torch.mean(fake_output)
            generator_losses.append(l)
            loss += l
        return loss, generator_losses

    def mse_mel_loss(self, fake_samples, real_samples):
        loss = 0
        for fake_sample, real_sample in zip(fake_samples,real_samples):
            l = self.mse_mel_loss_fn(fake_sample,real_sample)
            loss += l 
        return loss


    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss*2


    def calculate_gradient_penalty(self, real_data, fake_data, real_output, fake_output, k=2, p=6, device=torch.device("cpu")):
        real_gradient = torch.autograd.grad(
            outputs=real_output,
            inputs=real_data,
            grad_outputs=torch.ones_like(real_output),
            create_graph=True,
            retain_graph=True,
            # only_inputs=True,
        )[0]
        fake_gradient = torch.autograd.grad(
            outputs=fake_output,
            inputs=fake_data,
            grad_outputs=torch.ones_like(fake_output),
            create_graph=True,
            retain_graph=True,
            # only_inputs=True,
        )[0]
        real_gradient_norm = real_gradient.view(real_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
        fake_gradient_norm = fake_gradient.view(fake_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)

        gradient_penalty = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
        return gradient_penalty