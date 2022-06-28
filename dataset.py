from datasets import load_dataset, Audio
import datasets
import os

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import Data2VecAudioModel, Data2VecAudioConfig, Wav2Vec2Processor, AutoModel, AutoConfig

from datasets import ReadInstruction

class Timit(LightningDataModule):
    def __init__(
        self,
        # model_name_or_path: str,
        # max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        train_ratio: ReadInstruction = ReadInstruction(
                'train',from_=0, to=-1, unit='%'),
        val_ratio: ReadInstruction = ReadInstruction(
                'validation',from_=0, to=-1, unit='%'),
        test_ratio: ReadInstruction = ReadInstruction(
                'test',from_=0, to=-1, unit='%'),
        **kwarg,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.processor = Wav2Vec2Processor.from_pretrained(
            "./cache/models/data2vec")
        self.sampling_rate = 16000
        self.cache_dir = "./cache/datasets/timit_asr"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

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

    def setup(self, stage=None):
        self.splits = dict(
            train=load_dataset(self.cache_dir, split=self.train_ratio, data_dir=self.cache_dir, cache_dir=self.cache_dir),
            val=load_dataset(self.cache_dir, split=self.test_ratio, data_dir=self.cache_dir, cache_dir=self.cache_dir),
            test=load_dataset(self.cache_dir, split=self.test_ratio, data_dir=self.cache_dir, cache_dir=self.cache_dir)
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
        load_dataset(self.cache_dir, data_dir=self.cache_dir, cache_dir=self.cache_dir)

    def collate_fn(self, batch):
        processed = self.processor([it['input_values'] for it in batch], padding='max_length', max_length=256000, truncation=True,
                                   return_tensors="pt", batched=True, sampling_rate=16000)
        output = dict(
            input_values=processed['input_values'],
            attention_mask=processed['attention_mask'],
            input_length=processed['input_values'].shape[-1],
            batch_size=processed['input_values'].shape[0]
        )
        return output

    def train_dataloader(self):
        return DataLoader(self.splits['train'], batch_size=self.train_batch_size, collate_fn=self.collate_fn,shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.splits['val'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn,shuffle = True)

    def test_dataloader(self):
        return DataLoader(self.splits['test'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn,shuffle = True)


class LibriSpeech(LightningDataModule):
    def __init__(
        self,
        # model_name_or_path: str,
        # max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        train_ratio: ReadInstruction = ReadInstruction(
                'train.360',from_=0, to=-1, unit='%'),
        val_ratio: ReadInstruction = ReadInstruction(
                'validation',from_=0, to=-1, unit='%'),
        test_ratio: ReadInstruction = ReadInstruction(
                'test',from_=0, to=-1, unit='%'),
        streaming: bool = True, 
        # True if streaming, False if not streaming, noted that streaming is not compatible with loading ratio
        **kwarg,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.processor = Wav2Vec2Processor.from_pretrained(
            "./cache/models/data2vec")
        self.sampling_rate = 16000
        self.cache_dir = "./cache/datasets/LibriSpeech"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.streaming = streaming

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

    def setup(self, stage=None):
        self.splits = dict(
            train=load_dataset("./cache/datasets/LibriSpeech", 'clean', split=
                               self.train_ratio, cache_dir=self.cache_dir, streaming=self.streaming).with_format("torch"),
            val=load_dataset("./cache/datasets/LibriSpeech", 'clean', split=self.val_ratio,
                             cache_dir=self.cache_dir, streaming=self.streaming).with_format("torch"),
            test=load_dataset("./cache/datasets/LibriSpeech", 'clean', split=self.test_ratio,
                              cache_dir=self.cache_dir, streaming=self.streaming).with_format("torch")
        )
        # for k,v in self.splits.items():
        #     self.splits[k] = v.map(self.prepare_batch,remove_columns=v.column_names)
        # for k,v in self.splits.items():
        #     self.splits[k] = v.map(lambda x: self.processor(x,padding=True,pad_to_multiple_of = 320, return_tensors="pt"),batched=True)
        for k, v in self.splits.items():
            # breakpoint()
            self.splits[k] = v.map(self.get_array,  batched=False)
            # breakpoint()
        # for k, v in self.splits.items():
        #     print(self.splits[k])
        #     self.splits[k] = v.map(
        #         self.prepare_batch,  batched=True)

    def prepare_data(self):
        load_dataset(
            "./cache/datasets/LibriSpeech", 'clean', cache_dir=self.cache_dir)

    def collate_fn(self, batch):
        processed = self.processor([it['input_values'] for it in batch], padding='max_length', max_length=256000, truncation=True,
                                   return_tensors="pt", batched=True, sampling_rate=16000)
        output = dict(
            input_values=processed['input_values'],
            attention_mask=processed['attention_mask'],
            input_length=processed['input_values'].shape[-1],
            batch_size=processed['input_values'].shape[0]
        )
        return output

    def train_dataloader(self):
        return DataLoader(self.splits['train'], batch_size=self.train_batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.splits['val'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.splits['test'], batch_size=self.eval_batch_size, collate_fn=self.collate_fn)


if __name__ == "__main__":
    # dataset = Timit()
    # dataset.prepare_data()
    dataset = LibriSpeech()
    dataset.setup()
    d = dataset.val_dataloader()
    batch = next(iter(d))
    breakpoint()
    # dataset.prepare_data()
