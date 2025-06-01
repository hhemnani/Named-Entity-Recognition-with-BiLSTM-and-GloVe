#!/usr/bin/env python
# coding: utf-8

# <center><h1>NLP_HomeWork4_Task2</h1></center>
# <br>
# <br>

# Importing Libraries

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from collections import Counter
import random
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import gzip
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Reading Data for GloVe word embeddings
# 
# as we have to take care of upper case as well redoing everything

# In[2]:


#Found a better method to load data 
#Loading sequence data without labels from a file into a list of (words, tags) tuples
def load_data_to_dataframe(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    words, tags = [], []
    unique_words, unique_tags = set(), set()
    for line in lines:
        if line.strip() == "":
            data.append((words, tags))
            unique_words.update(words)
            unique_tags.update(tags)
            words, tags = [], []
        else:
            _, word, tag = line.strip().split()
            words.append(word)
            tags.append(tag)
    if words and tags:
        data.append((words, tags))
        unique_words.update(words)
        unique_tags.update(tags)

    return data, unique_words, unique_tags


def load_test_data_to_dataframe(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    words, tags = [], []
    for line in lines:
        if line.strip() == "":
            data.append((words, tags))
            words, tags = [], []
        else:
            _, word, tag = line.strip().split()
            words.append(word)
            tags.append(tag)
    if words and tags:
        data.append((words, tags))
    return data


# Preprocessing Data Using different functions and preprocessing techniques to get a better F1 score and handle Upper Cases as well
# not getting resonable F1 score so Trying to use "init" and "eos" tag to get better accuracy ref:https://arxiv.org/abs/1409.3215

# In[3]:


# CReating Mappings Considering Upper case letters as well
def case_sensitive_mappings(raw_data, unique_tags, threshold):
    word_freqs = Counter(word.lower() for words, _ in raw_data for word in words)
    filtered_words = [word.lower() for word, count in word_freqs.items() if count >= threshold]
    
    word_index = {word: idx + 4 for idx, word in enumerate(filtered_words)}
    word_index['<pad>'] = 0
    word_index['<s>'] = 1
    word_index['</s>'] = 2
    word_index['<unk>'] = 3

    tag_index = {tag: idx + 3 for idx, tag in enumerate(unique_tags)}
    tag_index['<pad>'] = 0
    tag_index['<s>'] = 1
    tag_index['</s>'] = 2

    return word_index, tag_index
##Pad Sequences for the tags
def pad_sequences(batch, word_index, tag_index, pad_token='<pad>', init='<s>', eos='</s>', unk='<unk>'):
    max_len = max([len(seq) + 2 for seq, _ in batch])  # Add 2 to account for <s> and </s> tokens

    padded_word_seqs = []
    padded_upper_seqs = []
    padded_tag_seqs = []

    for words, tags in batch:
        lower_words = [word.lower() for word in words]

        padded_words = [init] + lower_words + [eos]
        padded_words = [word_index.get(word, word_index[unk]) for word in padded_words] + [word_index[pad_token]] * (max_len - len(padded_words))
        padded_word_seqs.append(padded_words)

        padded_uppers = [0] + [int(word[0].isupper()) for word in words] + [0] + [0] * (max_len - len(words) - 2)
        padded_upper_seqs.append(padded_uppers)

        padded_tags = [init] + tags + [eos]
        padded_tags = [tag_index[tag] for tag in padded_tags] + [tag_index[pad_token]] * (max_len - len(padded_tags))
        padded_tag_seqs.append(padded_tags)

    return torch.tensor(padded_word_seqs), torch.tensor(padded_upper_seqs), torch.tensor(padded_tag_seqs)

def preprocess(text, word_index, pad_token='<pad>', init='<s>', eos='</s>', unk='<unk>'):
    tokens = text.split()

    lower_tokens = text.lower().split()
    padded_tokens = [init] + lower_tokens + [eos]
    indices = [word_index.get(word, word_index[unk]) for word in padded_tokens]
    
    upper_indices = [0] + [int(token[0].isupper()) for token in tokens] + [0]
    
    return indices, upper_indices


# Adding a Custom Dataset

# In[4]:


##Did not receive a good F1 score so Trying to Create Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[5]:


# Data Loaders and Mapping
train_file = "data/train" 
raw_data, unique_words, unique_tags = load_data_to_dataframe(train_file)
tokenized_data = [([word for word in words], [tag for tag in tags]) for words, tags in raw_data]
train_dataset = CustomDataset(tokenized_data)

dev_file = "data/dev" 
raw_data, unique_words, unique_tags = load_data_to_dataframe(dev_file)
tokenized_data = [([word for word in words], [tag for tag in tags]) for words, tags in raw_data]
dev_dataset = CustomDataset(tokenized_data)

word_index, tag_index = case_sensitive_mappings(raw_data, unique_tags, threshold=1)
train_loader = DataLoader(train_dataset,batch_size=8,collate_fn=lambda batch: pad_sequences(batch, word_index, tag_index),shuffle=True,)
dev_loader = DataLoader(dev_dataset,batch_size=8,collate_fn=lambda batch: pad_sequences(batch, word_index, tag_index),shuffle=True,)



# Sane Functions as Task 1

# In[6]:


def validate_with_metrics(model, dev_loader, loss_function, num_tags):
    model.eval()

    epoch_loss = 0
    y_true = []
    y_pred = []

    total_accuracy = 0
    total_amount = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dev_loader:
            word_seqs, upper_seqs, tag_seqs = batch
            word_seqs = word_seqs.to(device)
            upper_seqs = upper_seqs.to(device)
            tag_seqs = tag_seqs.to(device)

            logits = model(word_seqs, upper_seqs)
            logits = logits.view(-1, num_tags)
            tag_seqs = tag_seqs.view(-1)

            loss = loss_function(logits, tag_seqs)
            total_loss += loss.item()

            labels = tag_seqs.cpu().numpy()
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels)

            _, pred_tags = torch.max(logits, 1)
            y_pred.extend(pred_tags.cpu().numpy())
            # all_tags.extend(labels)

            mask = labels != 0
            correct_predictions = (predicted_labels[mask] == labels[mask]).sum()
            accuracy = correct_predictions / len(labels[mask])
            
            total_accuracy += accuracy
            epoch_loss += loss
            total_amount += 1

    precision, recall, f1_score, support = precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)

    print(f"Validation Loss: {(epoch_loss/total_amount)}, Accuracy: {(total_accuracy/total_amount)*100}%")
    print(f"Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1_score * 100:.2f}%")
    return (epoch_loss/total_amount), (total_accuracy/total_amount)*100, precision*100, recall*100, f1_score*100


# In[21]:


def predict_tags(model, input_text, word_to_index, idx2tag):
    model.eval()
    tokenized_input, upper_input = preprocess(input_text, word_to_index)
    input_tensor = torch.tensor([tokenized_input]).to(device)
    upper_tensor = torch.tensor([upper_input]).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor, upper_tensor)
    
    predicted_indices = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    predicted_tags = [idx2tag[idx] for idx in predicted_indices][1:-1]

    return predicted_tags


# In[8]:


# HYPER-PARAMETERS
vocab_size = len(word_index)
num_tags = len(tag_index)
embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128


# Load pre-trained GloVe embeddings from the gzip-compressed file
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

glove_file = "glove.6B.100d.gz"
glove_embeddings = load_glove_embeddings(glove_file)

embedding_matrix = np.zeros((vocab_size, 100)) 

for word, idx in word_index.items():
    if word in glove_embeddings:
        embedding_matrix[idx] = glove_embeddings[word]

embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)


# In[9]:


# Modifying BiLSTM model to use float32 data type for parameters
class BiLSTM_glove(nn.Module):
    def __init__(self, embedding_matrix, linear_output_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM_glove, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False).to(torch.float32)
        self.upper_embedding = nn.Embedding(2, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_output_dim, num_tags)

    def forward(self, x, upper_x):
        x = self.embedding(x)
        upper_x = self.upper_embedding(upper_x)
        x = torch.cat([x, upper_x], dim=-1)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        logits = self.linear2(x)

        return logits


# In[10]:


##Same as Task 1
def train_with_scheduler(model, train_loader, loss_function, optimizer, scheduler, num_epochs, clip_value, device, num_tags):

    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in train_loader:
            word_seqs, upper_seqs, tag_seqs = batch
            word_seqs, upper_seqs, tag_seqs = word_seqs.to(device), upper_seqs.to(device), tag_seqs.to(device)
            
            optimizer.zero_grad()
            logits = model(word_seqs, upper_seqs)
            logits = logits.view(-1, num_tags)
            tag_seqs = tag_seqs.view(-1)
            
            loss = loss_function(logits, tag_seqs)
            loss.backward()
            #Gradienr Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            
            total_loss += loss.item() * word_seqs.size(0)
            total_samples += word_seqs.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

def train_and_validate(model, train_loader, dev_loader, loss_function, optimizer, scheduler, num_epochs, clip_value, device, num_tags):
    best_f1_score = -1
    early_stopping_counter = 0
    patience = 5
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}")
        train_with_scheduler(model, train_loader, loss_function, optimizer, scheduler, 1, clip_value, device, num_tags)
        
        print(f"Validating Epoch {epoch + 1}/{num_epochs}")
        val_loss, val_accuracy, val_precision, val_recall, val_f1_score = validate_with_metrics(model, dev_loader, loss_function, num_tags)
        
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1_score/val", val_f1_score, epoch)
        
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            final_model=model
            print("updated")
            # early_stopping_counter = 0
            torch.save(model.state_dict(), "Blstm2.pt")
    writer.close()
    return final_model


# In[11]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_model = None
highest_f1_score = 0

# # Initializing the model with pre-trained embeddings
model = BiLSTM_glove(embedding_matrix, linear_output_dim, hidden_dim, num_layers, dropout)
model.to(device)

num_epochs = 25

loss_function = CrossEntropyLoss(ignore_index=tag_index['<pad>'])
optimizer = optim.SGD(model.parameters(), lr=0.15, momentum=0.9, weight_decay=0.00005)

patience = 5
writer = SummaryWriter()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, verbose=True)

best_f1_score = -1
clip_value = 5


# In[12]:


final_model=train_and_validate(model, train_loader, dev_loader, loss_function, optimizer, scheduler, num_epochs, clip_value, device, num_tags)


# In[23]:


#FUNCTION TO CREATE OUTPUT FILES
def save_predictions_dev(model, text_file, output_file, tag_to_index, word_to_index):
    with open(text_file, 'r') as input_file, open(output_file, 'w') as output_file:
        indices = []
        words = []
        tags = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0 and len(tags) > 0:
                    idx_to_tag = {idx: tag for tag, idx in tag_to_index.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags(model, new_text, word_to_index, idx_to_tag)

                    for i in range(len(indices)):
                        index = indices[i]
                        word = words[i]
                        tag = tags[i]
                        prediction = predicted_tags[i]

                        prediction_line = str(index) + " " + str(word) + " " + str(tag) + " " + str(prediction) + "\n"
                        output_file.write(prediction_line)

                    indices = []
                    words = []
                    tags = []
                    output_file.write("\n")
            else:
                index, word, tag = line.strip().split()
                indices.append(index)
                words.append(word)
                tags.append(tag)

def save_predictions_test(model, textFile, outputFile, tag2idx, word2idx):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags(model, new_text, word2idx, idx2tag)

                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        prediction = predicted_tags[i]

                        predictionLine = str(index) + " " + str(word) + " " + str(prediction) + "\n"
                        output_file.write(predictionLine)
                    
                    indexs = []
                    words = []
                    output_file.write("\n")
            else:
                index, word = line.strip().split()
                indexs.append(index)
                words.append(word)


# In[24]:


# CREATING OUTPUT FILES
save_predictions_dev(model, "data/dev", "dev2.out", tag_index, word_index)

save_predictions_test(model, "data/test", "test2.out", tag_index, word_index)

model.eval()


# In[ ]:





# In[ ]:




