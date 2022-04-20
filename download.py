
from models.SpeechAutoEncoder import SpeechAutoEncoder
from transformers import AutoModel, Data2VecAudioModel,Data2VecAudioConfig,AutoConfig,AutoModel,Wav2Vec2Processor


# encoder_config= Data2VecAudioConfig.from_pretrained("facebook/data2vec-audio-base-960h")
# decoder_config = AutoConfig.from_pretrained("jaketae/hifigan-lj-v1", trust_remote_code=True,        
#         upsample_rates=[2, 2, 2, 2, 2, 2, 5],
#         upsample_initial_channel=768,
#         upsample_kernel_sizes=[2, 2, 3, 3, 3, 3, 10],
#         model_in_dim=768,
#         sampling_rate=16000)
encoder = Data2VecAudioModel.from_pretrained('facebook/data2vec-audio-base-960h')

processor = Wav2Vec2Processor.from_pretrained(
            "facebook/data2vec-audio-base-960h")

decoder = AutoModel.from_pretrained("jaketae/hifigan-lj-v1", trust_remote_code=True)
# print(encoder_config)
# encoder_config.save_pretrained('./cache/configs/data2vec')
# decoder_config.save_pretrained('./cache/configs/hifigan/configuration_hifigan.json')
encoder.save_pretrained('./cache/models/data2vec')
processor.save_pretrained('./cache/models/data2vec')
decoder.save_pretrained('./cache/models/hifigan')

