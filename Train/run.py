import numpy as np
import pandas as pd
import time
import datetime
#import gc
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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
print(device)

df=pd.read_csv("output_en_200_curlie_original_trueconcat.csv",encoding="ISO-8859-1",on_bad_lines='skip', skiprows="")
#df.info()
df= df.sample(frac=1, random_state=42)
#htop
#df=df.iloc[:400]


df['Title'] = df['Title'].fillna("none")
df['Text'] = df['Text'].fillna("none")
df['Article'] = df['Article'].fillna("none")
df['Image_Name'] = df['Image_Name'].fillna("none")
df = df.dropna(subset=['is_dead'], axis=0)
df['is_dead'] = df['is_dead'].astype(bool)
# Convert non-numeric values to NaN
df['response_status_code'] = pd.to_numeric(df['response_status_code'], errors='coerce')

# Drop rows with NaN values in the 'response_status_code' column
df = df.dropna(subset=['response_status_code'])

# Convert 'response_status_code' column to integer
df['response_status_code'] = df['response_status_code'].astype(int)

import re

def sanitize_text(input_text):
    # Remove tabs, commas, and other special characters
    sanitized_text = re.sub(r'[\t,;:!\'"<>?~`@#$%^&*()\-_+=\[\]{}|\\\/]', ' ', input_text)
    sanitized_text=  sanitized_text.lower()
    
    # Replace multiple spaces with a single space
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text)
    
    return sanitized_text


# df['Text'] = df['Text'].apply(lambda x:sanitize_text(x))


#df['Concatenated'] = '[' + df['Url']+ ']' + '[' + df['Title'] + ']' + df['Text'] +df['Image_Name']  
#'[SEP]' #+df['Article']+
df['Title'] = df['Title'].apply(lambda x:sanitize_text(x))
df['Article'] = df['Article'].apply(lambda x:sanitize_text(x))
df['Image_Name'] = df['Image_Name'].apply(lambda x:sanitize_text(x))
df['Text'] = df['Text'].apply(lambda x:sanitize_text(x))
df['Concatenated'] = '[CLS]'  + df['Title'] +  '[SEP]'+df['Article']+ df['Text']+ df['Image_Name']

#df['Concatenated'] = df['Concatenated'].apply(lambda x:sanitize_text(x))
print(df)


df.loc[ df['response_status_code'] <= 205, 'response_status_code'] = int(0)
df.loc[df['response_status_code'] > 205 , 'response_status_code'] = int(1)

df.head()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
print("big")
df['Length'] = df['Concatenated'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))

# Drop rows where the length is greater than 450
df = df[df['Length'] <= 450]

Content = df.Concatenated.values
labels = df.response_status_code.values
#print(Content)
print (labels)
# Load the BERT tokenizer
print(' Original: ', Content[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(Content[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(Content[0])))

max_len = 0

# For every sentence...
for sent in Content:
    #print(sent)
   
    #Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))
    if len(input_ids) >450 :
        print(sent)

print('Max sentence length: ', max_len)

input_ids = []
attention_masks = []

# For every webpage...
for sent in Content:
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
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', Content[0])
print('Token IDs:', input_ids[0])

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a  train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.8 * len(dataset))
#val_size = int(0.4 * len(dataset))
val_size = len(dataset)  - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))



# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 64

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = True, # Whether the model returns attentions weights.
    output_hidden_states = True, # Whether the model returns all hidden-states.
)

# if device == "cuda:0":
# # Tell pytorch to run this model on the GPU.

#     model = model.cuda()
model = model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = 3e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

from tqdm import tqdm

from tqdm import tqdm

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(epochs):
    
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    total_train_loss = 0
    model.train()
    
    # Wrap the train_dataloader with tqdm for the progress bar
    train_iterator = tqdm(train_dataloader, desc="Training Iteration")
    
    for step, batch in enumerate(train_iterator):
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        optimizer.zero_grad()
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = output.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Update the tqdm progress bar description with the current loss
        train_iterator.set_description(f"Training Loss: {loss.item():.4f}")
    
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    best_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    # Wrap the validation_dataloader with tqdm for the progress bar
    val_iterator = tqdm(validation_dataloader, desc="Validation Iteration")
    threshold = 0.8
    for batch in val_iterator:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
        loss = output.loss
        total_eval_loss += loss.item()
        logits = output.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        # Update the tqdm progress bar description with the current loss
        predictions = []
        for probability in logits:
            if probability[1] >= threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        accuracy = np.mean(predictions == label_ids)
        precision, recall, f1, support = precision_recall_fscore_support(label_ids, predictions, average='binary')
        val_iterator.set_description(f"Validation Loss: {loss.item():.4f}, Threshold: {threshold}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
    
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    
    if avg_val_accuracy > best_eval_accuracy:
        torch.save(model, 'bert_model')
        best_eval_accuracy = avg_val_accuracy
    
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))



