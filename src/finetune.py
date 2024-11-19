
import transformers
from transformers import Trainer
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType


from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import os
import random

import fim

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B")

@dataclass
class DataArguments:
    data_column: str = field(default="content", metadata={"help": "Column name for the text content in the dataset"})
    data_path: str = field(default="bigcode/the-stack-smol", metadata={"help": "Path to the training data."})
    dataset_name: str = field(default="bigcode/the-stack-smol", metadata={"help": "Name of the dataset to use"})
    eos_token_id: Optional[int] = field(default=None, metadata={"help": "End of sequence token ID"})
    fim_rate: float = field(default=0.5, metadata={"help": "Rate (0.0 to 1.0) that sample will be permuted with FIM"})
    fim_spm_rate: float = field(default=0.5, metadata={"help": "Rate (0.0 to 1.0) of FIM permuations that will use SPM"})
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers for data loading"})
    seq_length: int = field(default=2048, metadata={"help": "Length of token sequences to return"})
    shuffle_buffer: int = field(default=5000, metadata={"help": "Size of the shuffle buffer"})
    size_valid_set: int = field(default=4000, metadata={"help": "Size of the validation set"})
    split: str = field(default="train", metadata={"help": "Split of the dataset to use"})
    streaming: bool = field(default=False, metadata={"help": "Whether to stream the dataset"})
    subset: str = field(default="data", metadata={"help": "Subset of the dataset to use, python, docker..."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


# 
# From [SantaCoder](https://github.com/loubnabnl/santacoder-finetuning.git)
#
def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else args.eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = fim.get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            np_rng = np.random.RandomState(seed=self.seed)
            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = fim.permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                        "input_ids": torch.LongTensor(example),
                        "labels": torch.LongTensor(example),
                    }

def create_datasets(tokenizer, args, seed):

    dataset = load_dataset(
        args.data_path,
        data_dir=args.subset,
        split=args.split,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.data_column)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=seed,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=seed,
    )

    return train_dataset, valid_dataset


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # https://github.com/huggingface/transformers/issues/33634
    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    # TODO use config for PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=target_modules, r=16, lora_alpha=16, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Load model from {} over.".format(model_args.model_name_or_path))

    train_dataset, eval_dataset = create_datasets(tokenizer, data_args, seed=training_args.data_seed)

    # print("Training dataset samples:", len(train_dataset))
    for index, sample in enumerate(train_dataset):
        print(f"Sample {index} of the training set: {sample['input_ids']}, {sample['labels']}.")
        print(f"Sample {index} of the training set: {tokenizer.decode(list(sample['input_ids']))}.")
        if index > 3:
            break

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )


    print("Training...")
    trainer.train()

    print("Training done")

    import pdb;pdb.set_trace()
    print("Saving last checkpoint of the model")

    tokenizer.save_pretrained(training_args.output_dir) 
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()