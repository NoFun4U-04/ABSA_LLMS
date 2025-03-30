import torch
import os
import argparse
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import default_data_collator, get_linear_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict

from utils import *
from eval_utils import *
from preprocessing import *
from init_trainer import *
from prompt_templates import *

parser = argparse.ArgumentParser()

parser.add_argument("--model_id", type=str, default="ura-hcmut/ura-llama-7b")
parser.add_argument("--domain", choices=["Restaurant", "Phone", "Education", "Hotel", "Mother", "Technology"], default="Restaurant")
parser.add_argument("--task", choices=["pair", "triplet", "quadruplet"])
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--prompt_format", choices=['SEALLM_V2_5', 'SEALLM_V2', 'SEALLM_V3'], default=None)
parser.add_argument("--model_type", choices=['seq2seq', 'causal'], default='seq2seq')
parser.add_argument("--add_instruction", action='store_true')
parser.add_argument("--using_trainer", action='store_true')
parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
parser.add_argument("--peft", action='store_true')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
#======================================
model_id = args.model_id
domain = args.domain
task = args.task
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
os.environ['domain'] = domain
r = args.lora_rank
model_type = args.model_type
add_instruction = args.add_instruction
using_trainer = args.using_trainer
gradient_accumulation_steps = args.gradient_accumulation_steps
if not gradient_accumulation_steps:
    gradient_accumulation_steps = 32 // batch_size
prompt_format = args.prompt_format
#======================================
print("="*50)
print("[INFO CUDA is Available: ",torch.cuda.is_available())
print("[INFO] Device: ", device)
print("[INFO] Model ID: ", model_id)
print("[INFO] Learning Rate: ", lr)
print("[INFO] Number of Epochs: ", num_epochs)
print("[INFO] Batch Size: ", batch_size)
print("[INFO] LoRA Rank:", r)
print("[INFO] Type of Model:", model_type)
print("[INFO] Using Instruction:", add_instruction)
print("[INFO] Using Trainer:", using_trainer)
print("[INFO] Gradient Accumulation Steps:", gradient_accumulation_steps)
print("[INFO] Prompt Format:", prompt_format)
print("[INFO] Add instruction:", add_instruction)
print("[INFO] Domain:", domain)
print("="*50)
#======================================

df_train, df_test = read_data(domain, task)

text_column = "input"
label_column = "output"
    
def create_instruction_input_output(df):
    input_text = []
    output_text = []
    for index, row in df.iterrows():
        input_review = clean_doc(row['input'], word_segment=False, max_length=512,lower_case=True)
        completion = row['output']
        prompt = get_prompt(input_review, prompt_format, task)
        
        prompt = prompt.replace("..", ".")
        if add_instruction:
            input_text.append(prompt)
        else:
            input_text.append(input_review)
        output_text.append(completion)
    print(len(input_text),len(output_text))
    return input_text,output_text

#======================================
# Train

input_train, output_train = create_instruction_input_output(df_train)

train_df = pd.DataFrame(list(zip(input_train, output_train)), columns =['text', 'label'])

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="models/")
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",cache_dir="models/") if model_type == 'causal' else AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto",cache_dir="models/")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


if model.config.architectures[0] == 'BloomForCausalLM':
    target_modules = [
                    "query_key_value",
                    "dense",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                ]
elif model.config.architectures[0] == 'LlamaForCausalLM':
    target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
else:
    target_modules = None

peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM if model_type == 'causal' else TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=r, # Lora attention dimension.
        lora_alpha=32, # the alpha parameter for Lora scaling.
        lora_dropout=0.05, # the dropout probability for Lora layers.
        target_modules=target_modules,
)

model = get_peft_model(model, peft_config) if args.peft else model

print_trainable_parameters(model)


max_input_length = max([len(tokenizer(text)['input_ids']) for text in input_train])
max_output_length = max([len(tokenizer(text)['input_ids']) for text in output_train])

tds = Dataset.from_pandas(train_df)

dataset = DatasetDict()
dataset['train'] = tds
print(dataset)

# data preprocessing
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model_type == 'causal':
    max_input_length = max([len(tokenizer(text + label)['input_ids']) for text, label in zip(input_train, output_train)])
else:
    max_input_length = max([len(tokenizer(text)['input_ids']) for text in input_train])
max_output_length = max([len(tokenizer(text)['input_ids']) for text in output_train])
print(max_input_length)
print(max_output_length)


def preprocess_function_for_causal_lm(examples):
    batch_size = len(examples["text"])
    inputs = [item + " " for item in examples["text"]]
    targets = examples["label"]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_input_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_input_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_input_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_input_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_input_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_input_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_for_seq2seq_lm(examples):
    inputs = tokenizer(text_target=examples['text'], max_length=max_input_length, padding='max_length', truncation=True)
    labels = tokenizer(text_target=examples['label'], max_length=max_output_length, padding='max_length', truncation=True)
    labels['input_ids'] = [
        [input_id if input_id != tokenizer.pad_token_id else -100 for input_id in input_ids] for input_ids in labels['input_ids']
    ]
    inputs['labels'] = labels['input_ids']
    assert len(inputs['labels']) == len(inputs['input_ids'])
    return inputs


processed_datasets = dataset.map(
    preprocess_function_for_causal_lm if model_type == 'causal' else preprocess_function_for_seq2seq_lm,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
print('='*50)
for i in range(5):
    example = train_dataset[i]
    print('[INFO] Raw text:',dataset['train']['text'][i])
    print('[INFO] Label:',dataset['train']['label'][i])
    for column in train_dataset.column_names:
        print(f"[INFO] {column}: {example[column]}")
    print(f"[INFO] Decoded Text: {tokenizer.decode(example['input_ids'])}")
print('='*50)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)


# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
if using_trainer:
    trainer = init_trainer(model, tokenizer, train_dataset, lr, batch_size, num_epochs, gradient_accumulation_steps)
else:
    trainer = None
    
import time
start_time= time.time() # set the time at which inference started

if using_trainer:
    trainer.train()
else:
# training and evaluation
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            #         print(batch)
            #         print(batch["input_ids"].shape)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

stop_time=time.time()
time_training =stop_time - start_time
print("Training time (seconds): ", time_training)

# Load dataset from the hub and get a sample
def get_prediction(example):
    input_review = clean_doc(example, word_segment=False, max_length=max_input_length, lower_case=True)
    prompt = get_prompt(input_review, prompt_format, task)
    prompt = prompt.replace("..", ".")
    prompt = prompt if add_instruction else input_review
    input_ids = tokenizer(prompt, max_length=max_input_length, return_tensors="pt", padding="max_length", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_output_length, eos_token_id=tokenizer.eos_token_id)
    if model_type == 'seq2seq':
        preds = outputs.detach().cpu().numpy()[0]
    else:
        preds = outputs[:, max_input_length:].detach().cpu().numpy()[0]
    label = tokenizer.decode(preds, skip_special_tokens=True)
    return label

input_test, output_test = create_instruction_input_output(df_test)

import time
start_time= time.time() # set the time at which inference started

y_pred = []
cnt = 0
for text in tqdm(input_test, desc='[INFO] Running inference...'):
    cnt += 1
    pred = get_prediction(text)
    y_pred.append(pred)
    if cnt % 5 == 0:
        print(pred)

stop_time=time.time()
inference_time =stop_time - start_time
print("Inference time (seconds): ", inference_time)

df = pd.DataFrame(list(zip(input_test, output_test, y_pred)),
               columns =['text','y_true', 'y_pred'])
df.to_csv(model_id.replace("/", "-") + domain +  ".csv",index=False)
df.head()

results = eval_absa(df.y_pred.tolist(), df.y_true.tolist())
scores = (
    f"Domain: {domain}\n"
    f"Add instruction: {add_instruction}\n"
    f"Accuracy: {results['acc']:.4f}\n"
    f"Precision: {results['precision']:.4f}\n"
    f"Recall: {results['recall']:.4f}\n"
    f"F1-score: {results['f1']:.4f}\n"
    f"Training time: {time_training:.4f}\n"
    f"Inference time: {inference_time:.4f}"
)

text_score = "Model: " + model_id + "\n" + scores + "\n\n"
with open('score.txt', 'a') as file:
    file.write(text_score)

print("Model: ", model_id)
print(scores)
