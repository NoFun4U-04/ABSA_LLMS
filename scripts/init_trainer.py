import os
import torch
from transformers import (default_data_collator,
                          DataCollatorForSeq2Seq,
                          DataCollatorForLanguageModeling,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup,
                          Seq2SeqTrainer,
                          Trainer,
                          Seq2SeqTrainingArguments,
                          TrainingArguments)



def init_trainer(model, tokenizer, tokenized_dataset, learning_rate, batch_size, num_epochs, gradient_accumulation_steps):
    data_collator = default_data_collator

    # Define training args
    training_args = TrainingArguments(
        output_dir="checkpoint",
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate, # higher learning rate
        num_train_epochs=num_epochs,
        logging_dir="checkpoint/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="epoch",
        gradient_accumulation_steps=gradient_accumulation_steps,
        push_to_hub=True,
        save_total_limit=1,
        hub_model_id=f"kietnt0603/{os.getenv('domain')}"
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    return trainer
