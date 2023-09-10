import numpy as np
import pandas as pd
import time
import datetime
import gc
import random
from nltk.corpus import stopwords
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

import re

def sanitize_text(text):
    # Remove common CSV delimiters (comma, semicolon, tab, pipe, colon, and space)
    sanitized_text = re.sub(r'[,\t;|: ]', ' ', text)
    # Replace multiple spaces with a single space
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text)
    sanitized_text=  sanitized_text.lower()
    # Split the text into words and take the first 100 words
    words = sanitized_text.split()[:100]
    # Join the first 100 words back into a single string
    sanitized_text = ' '.join(words)
    return sanitized_text


def sanitize_text_old(input_text):
    # Remove tabs, commas, and other special characters
    sanitized_text = re.sub(r'[\t,;:!\'"<>?~`@#$%^&*()\-_+=\[\]{}|\\\/]', ' ', input_text)
    
    # Replace multiple spaces with a single space
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text)
    
    return sanitized_text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dft=pd.read_csv("Traindf.csv")
dft.info()
dft= dft.sample(frac=1, random_state=42)


dft['Title'] = dft['Title'].fillna("none")
dft['Text'] = dft['Text'].fillna("none")
#dft['Article'] = dft['Article'].fillna("none")
dft['Image_Name'] = dft['Image_Name'].fillna("none")
#dft = dft.dropna(subset=['is_dead'], axis=0)
#dft['is_dead'] = dft['is_dead'].astype(bool)
# Convert non-numeric values to NaN
#dft['response_status_code'] = pd.to_numeric(dft['response_status_code'], errors='coerce')

# Drop rows with NaN values in the 'response_status_code' column
#dft = df.dropna(subset=['response_status_code'])

# Convert 'response_status_code' column to integer
#dft['response_status_code'] = dft['response_status_code'].astype(int)
dft['Title'] = dft['Title'].apply(lambda x:sanitize_text(x))
#dft['Article'] = dft['Article'].apply(lambda x:sanitize_text(x))   +dft['Article']+  '[SEP]'+
dft['Image_Name'] = dft['Image_Name'].apply(lambda x:sanitize_text(x))
dft['Text'] = dft['Text'].apply(lambda x:sanitize_text(x))
dft['Concatenated'] = '[CLS]'  + dft['Title'] +  '[SEP]'+ dft['Text']+  '[SEP]'+ dft['Image_Name']

dft['Length'] = dft['Concatenated'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))

# Drop rows where the length is greater than 450
dft = dft[dft['Length'] <= 450]

# Convert the values based on the condition
dft.loc[dft['response_status_code'] <= 200, 'response_status_code'] = 0
dft.loc[dft['response_status_code'] > 200, 'response_status_code'] = 1

# Check the unique values in the 'response_status_code' column
unique_values = dft['response_status_code'].unique()
print(unique_values)

Content1 = dft.Concatenated.values
labels1 = dft.response_status_code.values
print(Content1[0])
print ((labels1))

max_len = 0

# For every sentence...
print(len(Content1))
for sent in Content1:
    #print(sent)
   
    #Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))
    if len(input_ids) >450 :
        print(sent)

print('Max sentence length: ', max_len)

input_ids1 = []
attention_masks1 = []
asa=0
# For every webpage...
for sent in Content1:
    if asa%1000:
        print(asa)
    asa+=1
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        truncation=True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids1.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks1.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids1 = torch.cat(input_ids1, dim=0)
attention_masks1 = torch.cat(attention_masks1, dim=0)
labels1 = torch.tensor(labels1)
print(labels1)

# Print sentence 0, now as a list of IDs.
print('Original: ', len(Content1))
      
print('Token IDs:', input_ids1[0])

dataset = TensorDataset(input_ids1, attention_masks1, labels1)
model = torch.load('bert_model')
test_dataset = TensorDataset(input_ids1, attention_masks1)
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = 128 # Evaluate with this batch size.
        )

from tqdm import tqdm

predictions = []
# Wrap the test_dataloader with tqdm for the progress bar
test_iterator = tqdm(test_dataloader, desc="Testing Iteration")

for batch in test_iterator:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    with torch.no_grad():
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = output.logits
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        predictions.extend(list(pred_flat))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming you have prediction and label arrays
#predictions = np.array([1, 0, 0, 1, 1, 0])
#labels = np.array([1, 1, 0, 1, 0, 0])
from sklearn.metrics import accuracy_score, recall_score, precision_score


# Calculate accuracy
accuracy = accuracy_score(labels1,predictions)
print("Accuracy:", accuracy)

# Calculate recall
recall = recall_score(labels1, predictions)
print("Recall:", recall)

# Calculate precision
precision = precision_score(labels1, predictions)
print("Precision:", precision)


# Compute the confusion matrix
cm = confusion_matrix(labels1, predictions)

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(labels1)))
plt.xticks(tick_marks, np.unique(labels1))
plt.yticks(tick_marks, np.unique(labels1))
#plt.xticks(tick_marks, ['200 OK', '404 Error'])
#plt.yticks(tick_marks, ['200 OK', '404 Error'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

