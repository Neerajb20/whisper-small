import torch
from transformers import WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperProcessor
from datasets import load_dataset
import IPython.display as ipd
from datasets import Dataset, DatasetDict
import torchaudio
import torch
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
import os
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN1")
if HUGGINGFACE_TOKEN is None:
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable.")

from huggingface_hub import login
login(token=HUGGINGFACE_TOKEN)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "English"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained("openai/whisper-small",language="English", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
metric = evaluate.load("wer")
model.to("cuda:6") 

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

loaded_dataset = DatasetDict.load_from_disk("common_voice_processed")

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-en",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=loaded_dataset["train"],
    eval_dataset=loaded_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()