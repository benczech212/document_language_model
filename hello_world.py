from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained GPT-2 model
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load and tokenize your dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='/home/benczech/dev/document_language_model/data.txt',  # Replace with the path to your dataset file
    block_size=128,
)

# Check the size and content of the dataset
print(f"Number of lines in the dataset: {len(train_dataset)}")

if len(train_dataset) > 0:
    # Print the first few lines of the dataset
    print(f"First few lines of the dataset: {train_dataset[0]}")

    # Prepare for training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # No masked language modeling for GPT-2
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir='./finetuned_model',
        overwrite_output_dir=True,
        num_train_epochs=1,  # Adjust as needed
        per_device_train_batch_size=4,  # Adjust based on your GPU memory
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()
else:
    print("The dataset is empty. Please check the file path and content.")
