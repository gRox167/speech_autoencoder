from pathlib import Path
from torch import optim
from models.HiFiGAN_Generator import SpeechGenerator
from models.HiFiGAN_Discriminator import SpeechDiscriminator
from tasks.SpeechWGAN import SpeechWGAN
import torch
from transformers import AutoConfig
from dataset import Timit
from datasets import ReadInstruction

MODEL_NAME = 'SpeechWGAN_Timit'
loss = torch.nn.MSELoss()
task = SpeechWGAN
sampling_rate = 16000
base_path = Path(__file__).parent.parent

model_config = dict(
    encoder_config=None,
    pretrain_encoder_flag=True,
    decoder_config=AutoConfig.from_pretrained(base_path/"cache"/"models"/"hifigan", local_files_only=True, trust_remote_code=True,
                                              upsample_rates=[
                                                  2, 2, 2, 2, 2, 2, 5],
                                              upsample_initial_channel=768,
                                              upsample_kernel_sizes=[
                                                  2, 2, 3, 3, 3, 3, 10],
                                              model_in_dim=768,
                                              sampling_rate=sampling_rate),
)

task_config = dict(
    generator=SpeechGenerator(**model_config),
    discriminator=SpeechDiscriminator(),
    optimizer=optim.Adam,
    n_critics=5,
    loss_fn=loss,
    lr=[0.00001, 0.00001],
    sampling_rate=sampling_rate,
)

dataset_config = dict(
    dataset_class = Timit,
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=40,
    train_ratio= ReadInstruction(
                'train',from_=1, to=-1, unit='abs'),
    val_ratio=ReadInstruction(
                'validation',from_=1, to=-1, unit='abs'),
    test_ratio=ReadInstruction(
                'test',from_=1, to=16, unit='abs'),
    streaming=False,  # True if streaming, False if not streaming, noted that streaming is not compatible with loading ratio
)

train_config = dict(
    devices=[0,1,2,3],
    auto_select_gpus=True,
    mode='train',
    log_path='./experiments/'+MODEL_NAME,
    max_epochs=2000,
    # train_batch_size= 2,
    # eval_batch_size = 1,
    deterministic=False,
    precision=32,
    # strategy="deepspeed_stage_2"
    strategy="dp"
)
