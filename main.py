from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments

def train_model(train_csv="copy-train.csv", eval_csv="copy-eval.csv"):
    """Trains a SetFit model to identify gender assertions in news extracts for a given name."""

    # Load the datasets, and combine name and sentence into a single text field
    train_dataset = load_dataset("csv", data_files=train_csv)["train"].map(lambda x: {"text": f"{x['name']}: {x['sentence']}", "label": x["label"]})
    eval_dataset = load_dataset("csv", data_files=eval_csv)["train"].map(lambda x: {"text": f"{x['name']}: {x['sentence']}", "label": x["label"]})

    # Define the gender assertion labels
    labels = ["nonbinary", "binary", "unknown"]

    # Load the SetFit model - BAAI/bge-small-en-v1.5, w601sxs/b1ade-embed, mixedbread-ai/mxbai-embed-large-v1, Alibaba-NLP/gte-large-en-v1.5
    model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5", labels=labels)
    
    # Prepare training arguments
    args = TrainingArguments(
        batch_size=32,
        num_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Train the model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"}
        # compute_metrics=lambda p: {"accuracy": evaluate.load("accuracy").compute(predictions=p.predictions, references=p.label_ids)},
    )
    trainer.train()
    
    # Save the newly trained model
    model.save_pretrained("nonbinary-gender-estimator")

def use_model(data, model_path="nonbinary-gender-estimator"):
   """Uses a trained SetFit model to predict gender assertions in news extracts for a given name."""

   # Load the trained model
   model = SetFitModel.from_pretrained(model_path)

   # Format the data as required
   formatted_data = [f"{item['name']}: {item['sentence']}" for item in data]

   # Make predictions
   return model.predict(formatted_data)

def push_model(model_path="nonbinary-gender-estimator"):
   """Pushes the model to the Hub"""

   # Load the trained model
   model = SetFitModel.from_pretrained(model_path)

   # Push the model to the Hub
   model.push_to_hub("lujstn/nonbinary-gender-estimator")

def test_model(test_cases=None, model_path="nonbinary-gender-estimator"):
   """Tests the model with a set of test cases"""

   # Define the default test cases if none are provided
   if test_cases is None:
       test_cases = [
           ("Alex Morgan", "Alex Morgan, the renowned author, shared their thoughts on the new book release. They mentioned that the book was a reflection of their personal journey.", "nonbinary"),
           ("Alex Morgan", "Alex Morgan, the renowned author, shared his thoughts on the new book release. He mentioned that the book was a reflection of his personal journey.", "binary"),
           ("Alex Morgan", "Alex Morgan, the renowned author, shared thoughts on the new book release. The book was a reflection of a personal journey.", "unknown"),
           ("Jordan Lee", "Jordan Lee, a prominent scientist, presented their findings at the conference. They emphasized the importance of the research.", "nonbinary"),
           ("Jordan Lee", "Jordan Lee, a prominent scientist, presented her findings at the conference. He emphasized the importance of the research.", "binary"),
           ("Jordan Lee", "Jordan Lee, a prominent scientist, presented new findings at the conference. The importance of the research was emphasized.", "unknown")
       ]
   
   # Make use_model() function calls for each test case, and log the results
   for name, sentence, expected in test_cases:
      result = use_model([{"name": name, "sentence": sentence}], model_path)
      if result == expected:
         print(f"✅ PASS: {name} - {sentence} -> {result}")
      else:
         print(f"⛔ FAIL: {name} - {sentence} -> {result} (should have been {expected})")


# To use the model, use the following function call:
test_model([
   ("Aisha Khan", "Aisha Khan discussed the latest advancements in renewable energy at the environmental summit.", "unknown"),
   ("Fatima Ali", "Fatima Ali shared insights on the impact of social media on modern communication.", "unknown"),
   ("Nadia Hassan", "Nadia Hassan presented research findings on climate change at the global conference.", "unknown"),
   ("Elena Petrova", "Elena Petrova discussed her latest research on climate change. She emphasized the need for immediate action.", "binary"),
   ("Carlos Silva", "Carlos Silva shared his experiences as a scientist. He highlighted the importance of research.", "binary"),
   ("Jordan Taylor", "Jordan Taylor, a nonbinary activist, expressed their support for the new policy – who believes it will bring positive change.", "nonbinary"),
   ("Jordan Taylor", "Jordan Taylor, an activist, expressed their support for the new policy. They believe it will bring positive change.", "nonbinary"),
   ("Amina Yusuf", "Amina Yusuf and Rafael Torres presented their findings on renewable energy. Amina was leading the charge – he provided insights on the project's impact.", "binary"),
   ("Fatima Khan", "Fatima Khan, Aisha Rahman, and Javier Morales discussed their collaborative project. Fatima is the project organiser. She shared her perspective on the challenges faced The others did not - they pursued goals together", "binary")
])

# ⚠️ To train the model, uncomment the following line:
# train_model()

# ⚠️ To test the model, uncomment the following line:
# test_model()

# ⚠️ If you've tested the model and are happy with its predictions, uncomment to push the model to the Hub:
# push_model()