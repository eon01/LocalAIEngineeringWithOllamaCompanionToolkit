from unsloth import FastLanguageModel

# Reload the base model with your trained adapter on top.
# We point at the adapter folder, not the original model name.
# Unsloth reads the config inside it, fetches the matching base model,
# and reattaches your trained adapter automatically.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "granite_sql_lora",   # the folder you saved your adapter in
    max_seq_length = 2048,             # same length used during training
    load_in_4bit = True,               # load the compressed base, as in training
)

# Merge the adapter into the base and convert to GGUF in one call.
model.save_pretrained_gguf(
    "granite_sql_gguf",                # output folder for the .gguf file
    tokenizer,
    quantization_method = "q4_k_m",    # good balance of file size and quality
)
