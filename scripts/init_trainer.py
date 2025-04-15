import os
import torch
from transformers import (default_data_collator,
                          DataCollatorForLanguageModeling,
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup,
                          Trainer,
                          TrainingArguments)


def init_trainer(model, tokenizer, tokenized_dataset, learning_rate, batch_size, num_epochs, gradient_accumulation_steps):
    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir="./results",
        run_name="meta-llama/Llama-3.2-3B-Instruct",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=learning_rate,
        fp16=True,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    return trainer
