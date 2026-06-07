from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# How long each training example can be, in tokens.
# A SQL schema plus a question fits comfortably in 2048.
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/granite-4.0-micro",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # load the compressed 4-bit form (the QLoRA part)
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # adapter size
    lora_alpha=16,  # adapter strength
    lora_dropout=0,
    bias="none",
    target_modules=[  # where the adapter attaches (leave as-is)
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # saves memory during training
    random_state=3407,  # makes the run repeatable
)

dataset = load_dataset(
    "gretelai/synthetic_text_to_sql",
    split="train[:5000]",
)

# Confirm the column names before mapping them. A renamed column
# is a silent break, so check the real data once.
print(dataset.column_names)
print(dataset[0])

SYSTEM = (
    "You translate questions into SQL for the given schema. "
    "Reply with one SQL query and nothing else."
)


def format_example(row):
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Schema:\n{row['sql_context']}\n\nQuestion: {row['sql_prompt']}",
        },
        {"role": "assistant", "content": row["sql"]},
    ]
    # tokenize=False returns plain text; the trainer tokenizes later.
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


# Turn every row into a single "text" field the trainer reads.
dataset = dataset.map(format_example)

# Look at one formatted row to see the template Granite expects.
print(dataset[0]["text"])


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch size = 2 x 4 = 8
        warmup_steps=5,
        num_train_epochs=1,  # one full pass over the data
        learning_rate=2e-4,
        logging_steps=10,  # print loss every 10 steps
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)


# The role markers are Granite-specific; they mark where the
# user turn ends and the assistant answer begins.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_of_role|>user<|end_of_role|>",
    response_part="<|start_of_role|>assistant<|end_of_role|>",
)

trainer.train()

model.save_pretrained("granite_sql_lora")
tokenizer.save_pretrained("granite_sql_lora")
