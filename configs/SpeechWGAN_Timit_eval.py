from pathlib import Path
from torch import optim
from models.HiFiGAN_Generator import SpeechGenerator
from models.HiFiGAN_Discriminator import SpeechDiscriminator
from tasks.SpeechWGAN import SpeechWGAN
import torch
from transformers import Data2VecAudioConfig, AutoConfig


MODEL_NAME = 'SpeechWGAN_Timit'
loss = torch.nn.MSELoss()
task = SpeechWGAN
sampling_rate = 16000
base_path = Path(__file__).parent.parent
# print(base_path)
# print(base_path/"cache"/"configs"/"hifigan"/"config.json")

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

# dataset_config = dict(
#     dataset_class = Timit,
#     train_batch_size=1,
#     eval_batch_size=1,
#     num_workers=40,
#     train_ratio= ReadInstruction(
#                 'train',from_=1, to=-1, unit='abs'),
#     val_ratio=ReadInstruction(
#                 'validation',from_=1, to=50, unit='%'),
#     test_ratio=ReadInstruction(
#                 'test',from_=1, to=16, unit='abs'),
#     streaming=False,  # True if streaming, False if not streaming, noted that streaming is not compatible with loading ratio
# )

train_config = dict(
    devices=1,
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
# nohup python train.py --config config.segment_NIKENet_2D --mode continue --gpu 0 --checkpoint experiment/segment_NIKENetSeg_2D/last.ckpt > seg.out &
# nohup python train.py --config config.segment_NIKENet_2D --mode train --gpu 1&
# train ['887', '388', '429', '438', '125', '403', '674', '211', '432', '576', '732', '789', '723', '910', '891', '556', '801', '673', '379', '470']
# val ['481', '206', '551', '196', '176', '530']
