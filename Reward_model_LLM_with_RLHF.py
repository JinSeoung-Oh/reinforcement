### https://python.plainenglisgh.io/building-a-reward-model-for-your-llm-using-rlhf-in-python-49abaf4906f

## Step 1. Argilla server setup
# https://docs.argilla.io/en/latest/getting_started/quickstart.html

## Step 2. Install required packages
# pip install -U argilla pandas trl plotly -qqq

## Step 3. Import Necessary Libraies
import random
import torch
from datasets import Dataset, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments)
from trl import RewardTrainer
import argilla as rg

## Step 4. Initialize Argilla Client (optional)
rg.init(api_url = "http://localhost:    ", # Replace with your Argilla server's URL
        api_key = "admin.apikey")          # Replace with your API key if applicable

## Step 5. Load the Dataset
# In this example,
hf_dataset = load_dataset("argilla/dolly-curated-comparison-falcon-7b-instruct", split="train")

## Step 6. Convert to Pandas DataFrame
df = hf_dataset.to_pandas()

## Step 7. Define Fields for Dataset Settings
# In this example, instruction : corresponds to the user's instruction or prompt / response-1 : Represents first response / response-2 : Represents the second response
fields = [
  rg.TextField(name="instruction", title="User instruction"),
  rg.TextField(name="response-1"),
  rg.TextField(name="response-2")]

## Step 8. Configure the Labeling Question
question = rg.RatingQuestion(
  name = "choose-best",
  title = "CHoose the best response:",
  description = "Choose the most helpful, harmless, and truthful response. Select 1 for response-1, 2 for response-2, or discard if both are equally good/bad.",
  values = [1,2],
  required = True)
# name : Assigns a name to this question
# title : Specifies the question's title, which is displayed to lablers
# description : Provides detailed instructions to lablelers
# values : Indicates the possible values that labelers can choose
# required : Makes it mandatory for labelers to answer this question

## step 9. Provide Annotation Guidelines
guidline = """Theses Gudelines ... """

## step 10. Budilding Comparison Records
# comparison records to collect data
records = [rg.FeedbackRecord(fields={"instruction": r["prompt"], "response-1":r["original_response"], "response-2":r["response-2"]}) for r in hf_dataset]

## step 11. Creating a Dataset Configuration
dataset = rg.FeedbackDataset(fields = fields, question = [question], guidelines = guideline)

## step 12. Adding Records and Publishing the Dataset
dataset.add_records(records)
dataset.push_to_argilla(name="comparison-data-falcon")

## step 13. (optional) Pushing to Hugging Face Hub
dataset.push_to_huggingface("comparison-data-falcon")

##step 14. Retrieve Labeled Dataset
# If you have not labeled any responsese yet, download from hugging face
feedback_dataset = rg.FeedbackDataset.from_huggingface("argila/comparison-data-falcon-with-feedback")
# If you have labeled some responses in the UI, you can retrieve the labeled dataset from argilla
feedback_dataset = rg.FeedbackDataset.from_argilla('comparison-data-falcon')

## Step 15. Prepare the Dataset for Reward Modeling
from typing import Any, Dict
from argilla.feedback import TrainingTask
from collections import Counter

def formatting_func(sample:Dict[str, Any]):
    values = [
      annotation["value"]
      for annotation in sample["choose-best"]
      if annotation["status"] == "submitted"]
    winning_response = Counter(values).most_common(1)[0][0]
    if winning_response == 1:
        chosen = sample["response-1"]
        rejected = sample["response-2"]
    else:
         choosen = sample["response-2"]
         rejected = sample["response-1"]
    return chosen, rejected

task = TraininTask.for_reward_modeling(formatting_func=formatting_func)

## Step 16. Observe the Resulting Datset
dataset = feedback_dataset.prepare_for_training(framework="trl", task=task)
dataset

##### outputs starts #####
Dataset({features:['chosen', 'rejected'], num_rows=7401})
##### output ends #####

## Step 17. Choose a base model
model_name = "distilroberta-base"  #enter base model name

## Step 18. Initialize the Argilla Trainer
trainer = ArgillaTrainer(dataset = feedback_dataset, task=task, framwork="trl", model = model_name, train_size = 0.8,)

## Step 19. update Training configuration
trainer.update_config(per_device_train_batch_size = 16, evaluation_strategy="steps", logging_step=200,)

## Step 20. Start Training
trainer.train("./reward_model")  # ???

## Step 21. Load the Tokenizer and Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("argilla/roberta-base-reward-model-falcon-dolly")
model = AutoModelForSequenceClassification.from_pretrained("argilla/roberta-base-reward-model-falcon-dolly")

## step 22. Define a function to get the score
def get_score(model, tokenizer, prompt, response):
   inputs = tokenizer.encode_plus(prompt, response, truncation=True, padding="max_length", max_length=512, return_tensor="pt")
   with torch.no_grad():
      outputs = model(**inputs)
   logits = outputs.logits

   return logits.item()

## step 23. Use the reward model
prompt = "   "
example_less_pref_response = "   "
example_preferred_response = "    "

score_less_pref = get_score(model, tokenizer, prompt, example_less_pref_response)
print(score_less_pref)

score_preferred = get_score(model, tokenizer, prompt, example_preferred_response)
print(score_preferrd
