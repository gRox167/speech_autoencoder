from datasets import load_dataset, Audio
import datasets
import os

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import Data2VecAudioModel,Data2VecAudioConfig,Wav2Vec2Processor,AutoModel,AutoConfig

class Timit(LightningDataModule):
    def __init__(
        self,
        # model_name_or_path: str,
        # max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwarg,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.processor = Wav2Vec2Processor.from_pretrained("./cache/models/data2vec")
        self.sampling_rate = 16000

    def get_array(self, batch):
        audio = batch["audio"]
        batch['input_values'] = audio['array']
        batch["input_length"] = len(batch["input_values"])
        return batch

    # def prepare_batch(self, batch):
    #     array = batch["input_values"]
    #     # print(array)
    #     batch = self.processor(
    #         array, sampling_rate=self.sampling_rate, padding=True, pad_to_multiple_of=320, return_tensors="np")
    #     batch["input_length"] = batch["input_values"].shape[-1]
    #     return batch

    def setup(self,stage = None):
        self.splits = dict(
            train = load_dataset("timit_asr",split=datasets.ReadInstruction('train', to=95, unit='%'), cache_dir="./cache/datasets"),
            val = load_dataset("timit_asr",split=datasets.ReadInstruction('train', from_=-5, unit='%'), cache_dir="./cache/datasets"),
            test=load_dataset("timit_asr", split=datasets.ReadInstruction(
                'test', to=50, unit='%'), cache_dir="./cache/datasets")
        )
        # for k,v in self.splits.items():
        #     self.splits[k] = v.map(self.prepare_batch,remove_columns=v.column_names)
        # for k,v in self.splits.items():
        #     self.splits[k] = v.map(lambda x: self.processor(x,padding=True,pad_to_multiple_of = 320, return_tensors="pt"),batched=True)
        for k, v in self.splits.items():
            self.splits[k] = v.map(
                self.get_array, remove_columns=v.column_names, batched=False)
        # for k, v in self.splits.items():
        #     print(self.splits[k])
        #     self.splits[k] = v.map(
        #         self.prepare_batch,  batched=True)

    def prepare_data(self):
        load_dataset("timit_asr", cache_dir="./cache/datasets")

    def collate_fn(self, batch):
        processed = self.processor([it['input_values'] for it in batch], padding=True,
                                        pad_to_multiple_of=320, return_tensors="pt", batched=True,sampling_rate = 16000)
        output = dict(
            input_values=processed['input_values'],
            attention_mask=processed['attention_mask'],
            input_length=processed['input_values'].shape[-1]
        )
        return output

    def train_dataloader(self):
        return DataLoader(self.splits['train'], batch_size=self.train_batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.splits['val'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.splits['test'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

class LibriSpeech(LightningDataModule):
    def __init__(
        self,
        # model_name_or_path: str,
        # max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwarg,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/data2vec-audio-base-960h", cache_dir='./cache/models/data2vec')
        self.sampling_rate = 16000

    def get_array(self, batch):
        audio = batch["audio"]
        batch['input_values'] = audio['array']
        batch["input_length"] = len(batch["input_values"])
        return batch

    # def prepare_batch(self, batch):
    #     array = batch["input_values"]
    #     # print(array)
    #     batch = self.processor(
    #         array, sampling_rate=self.sampling_rate, padding=True, pad_to_multiple_of=320, return_tensors="np")
    #     batch["input_length"] = batch["input_values"].shape[-1]
    #     return batch

    def setup(self,stage = None):
        self.splits = dict(
            train = load_dataset("librispeech_asr",'clean',split='train.360', cache_dir="./cache/datasets/librispeech"),
            val = load_dataset("librispeech_asr",'clean',split='validation', cache_dir="./cache/datasets/librispeech"),
            test=load_dataset("librispeech_asr",'clean', split='test', cache_dir="./cache/datasets/librispeech")
        )
        # for k,v in self.splits.items():
        #     self.splits[k] = v.map(self.prepare_batch,remove_columns=v.column_names)
        # for k,v in self.splits.items():
        #     self.splits[k] = v.map(lambda x: self.processor(x,padding=True,pad_to_multiple_of = 320, return_tensors="pt"),batched=True)
        for k, v in self.splits.items():
            self.splits[k] = v.map(
                self.get_array, remove_columns=v.column_names, batched=False)
        # for k, v in self.splits.items():
        #     print(self.splits[k])
        #     self.splits[k] = v.map(
        #         self.prepare_batch,  batched=True)

    def prepare_data(self):
        # train = load_dataset("librispeech_asr", 'clean',cache_dir="./cache/datasets/librispeech"),
        pass

    def collate_fn(self, batch):
        processed = self.processor([it['input_values'] for it in batch], padding=True,
                                        pad_to_multiple_of=320, return_tensors="pt", batched=True,sampling_rate = 16000)
        output = dict(
            input_values=processed['input_values'],
            attention_mask=processed['attention_mask'],
            input_length=processed['input_values'].shape[-1]
        )
        return output

    def train_dataloader(self):
        return DataLoader(self.splits['train'], batch_size=self.train_batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.splits['val'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.splits['test'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn)