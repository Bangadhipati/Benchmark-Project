# Step 5: Train & Optimize Google Gemma-2-2B-IT (4-bit QLoRA)
HF_TOKEN = '' # HuggingFace Token for gated models
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
SAVE_STEPS = 100

import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

compute_dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
device_map = {'': 0} if torch.cuda.is_available() else None

train_df = pd.read_csv(TRAIN_CSV).dropna(subset=['Abstract', 'Highlights'])
val_df = pd.read_csv(VAL_CSV).dropna(subset=['Abstract', 'Highlights'])

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it', token=HF_TOKEN or None, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Preprocess with Strict Token Limits
def prep(batch):
    texts = [f'<start_of_turn>user\nGenerate concise scientific research highlights for this abstract:\n{a.strip()}<end_of_turn>\n<start_of_turn>model\n{h.strip()}<end_of_turn>' for a, h in zip(batch['Abstract'], batch['Highlights'])]
    tok = tokenizer(texts, max_length=MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH, truncation=True)
    tok['labels'] = tok['input_ids'].copy()
    return tok
raw_train_ds = Dataset.from_pandas(train_df, preserve_index=False)
train_ds = raw_train_ds.map(prep, batched=True, remove_columns=raw_train_ds.column_names)
raw_val_ds = Dataset.from_pandas(val_df, preserve_index=False)
val_ds = raw_val_ds.map(prep, batched=True, remove_columns=raw_val_ds.column_names)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2-2b-it',
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    token=HF_TOKEN or None
)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    bias='none'
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=MODEL_CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-4,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    save_strategy='steps',
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    fp16=True if not torch.cuda.is_bf16_supported() else False,
    bf16=True if torch.cuda.is_bf16_supported() else False,
    optim='paged_adamw_8bit' if True else 'adamw_torch',
    report_to='none'
)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=data_collator)
# Checkpoint Resuming Check
resume_ckpt = None
if os.path.exists(MODEL_CHECKPOINT_DIR):
    ckpts = [os.path.join(MODEL_CHECKPOINT_DIR, d) for d in os.listdir(MODEL_CHECKPOINT_DIR) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(MODEL_CHECKPOINT_DIR, d))]
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split('-')[-1]))
        resume_ckpt = ckpts[-1]
        print('--> 🔄 Found previous checkpoint on Google Drive! Resuming from:', resume_ckpt)

print('Starting training...')
trainer.train(resume_from_checkpoint=resume_ckpt)

final_adapter = os.path.join(MODEL_CHECKPOINT_DIR, 'final_adapter')
trainer.save_model(final_adapter)
tokenizer.save_pretrained(final_adapter)
print('✅ Fine-tuned adapter saved to Drive:', final_adapter)

del model, trainer
torch.cuda.empty_cache()
gc.collect()
