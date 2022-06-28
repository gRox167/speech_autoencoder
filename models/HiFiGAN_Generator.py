import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from transformers import Data2VecAudioModel,AutoConfig,PreTrainedModel


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class HiFiGAN(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(
                config.model_in_dim,
                config.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        config.upsample_initial_channel // (2**i),
                        config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=0,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, output_hidden_states=False):
        hidden_states = []
        x = self.conv_pre(x)
        # print(x.shape)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            hidden_states.append(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        hidden_states.append(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        hidden_states.append(x)
        if output_hidden_states:
            return x,hidden_states
        else:
            return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class SpeechGenerator(nn.Module):
    def __init__(self, encoder_config, pretrain_encoder_flag, decoder_config, output_hidden_states = False, **kwargs):
        super().__init__()
        self.encoder = Data2VecAudioModel.from_pretrained(
            './cache/models/data2vec') if pretrain_encoder_flag == True else Data2VecAudioModel(encoder_config)
        self.decoder = HiFiGAN(decoder_config)

    def forward(self, x, output_hidden_states=False):
        if output_hidden_states:
            x = self.encoder(x, output_hidden_states=True)
            encoder_hidden_states = x.hidden_states
            x, decoder_hidden_states = self.decoder(x.last_hidden_state.transpose(-1, -2),output_hidden_states)
            return x, encoder_hidden_states,decoder_hidden_states
        else:
            x = self.encoder(x)
            x = self.decoder(x.last_hidden_state.transpose(-1, -2),output_hidden_states)
            return x
        # output of encoder shape is like (batch,length,channel)
        # input of decoder shape is like (batch,channel,length)

if __name__=="__main__":
    decoder_configuration = AutoConfig.from_pretrained("jaketae/hifigan-lj-v1", trust_remote_code=True,        
        upsample_rates=[2, 2, 2, 2, 2, 2, 5],
        upsample_initial_channel=1536,
        upsample_kernel_sizes=[2, 2, 3, 3, 3, 3, 10],
        model_in_dim=768,
        sampling_rate=16000)
    decoder = HiFiGAN(decoder_configuration)
    with torch.no_grad():
        outputs_1 = decoder(torch.rand((1,158,768)))
    print(outputs_1.shape)

