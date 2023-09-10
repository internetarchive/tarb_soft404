import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer,BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


model = torch.load("bert_model1")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
INPUTFILE=""
OUTPUTFILE=""
#nltk.download('wordnet')#download it once should comment it or subsequent usage 


def make_prediction_Bert(INPUTFILE):
     
    with open('output1.csv', "r", encoding="cp1252", errors="replace") as f:
            df = pd.read_csv(f)

    #df=pd.read_csv('text1_bert_404.csv',encoding='cp1252')
    text=df['Text']
    #text = ' '.join(text.split()[:350])
    input_text = '[CLS]'  + df['Title'] + '[SEP]' + text +'[SEP]'+df['Article']+'[SEP]'+' '.join(str(df['Image_Name']))
    #input_text = ' '.join(input_text.split()[:450])

    df['input_text']=input_text
    df = df.dropna(subset=['input_text'])
    Content1 = df.input_text.values

    input_ids1 = []
    attention_masks1 = []

    for sent in Content1:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
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
    batch_size=16

    test_dataset = TensorDataset(input_ids1, attention_masks1)
    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )


    predictions = []
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

    # print(predictions)
    
    OUTPUTFILE=INPUTFILE+"_processed_Bert.csv"
    df['bert_prediction'] = predictions
    df1=df[['Url','bert_prediction','response_status_code']]
    df1.to_csv(OUTPUTFILE, index=False, encoding='cp1252')


def main():
    INPUTFILE=sys.argv[1]
    make_prediction_Bert(INPUTFILE)    


if __name__ == "__main__":
    main()

