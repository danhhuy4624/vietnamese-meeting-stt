import re
import unicodedata
from dataclasses import dataclass

import evaluate
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_ID = "vinai/PhoWhisper-small"
DATASET_ID = "hataphu/common-voice-corpus-20"
OUTPUT_DIR = "./models/phowhisper-small-meetings"

ds = load_dataset(DATASET_ID)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

processor = WhisperProcessor.from_pretrained(
    MODEL_ID,
    language="vi",
    task="transcribe",
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.generation_config.language = "vi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.config.use_cache = False


def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(example):
    audio = example["audio"]
    features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    )
    example["input_features"] = features.input_features[0]
    example["labels"] = processor.tokenizer(clean_text(example["sentence"])).input_ids
    return example


ds = ds.map(
    prepare,
    remove_columns=ds["train"].column_names,
    num_proc=4,
)


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [
            {"input_features": item["input_features"]}
            for item in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            return_attention_mask=True,
        )

        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


wer = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    predictions = processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )
    references = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True
    )
    return {"wer": 100 * wer.compute(predictions=predictions, references=references)}


args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    num_train_epochs=2,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    report_to="none",
    ddp_find_unused_parameters=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    ),
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
