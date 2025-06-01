#!/usr/bin/env python
# coding: utf-8

# <center><h1>NLP_Homework4_Task1</h1></center>
# <br>
# <br>

# Simple Bidirectional LSTM model

# In[3]:


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


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


##reading files line by line and extracting word-level information and 
##storing sentence IDs, word indices, words, and NER tags in a DataFrame.
    
def load_data_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as f:
        sentence_id = 0
        for line in f:
            if line.strip() == "":
                sentence_id += 1
            else:
                parts = line.strip().split()
                data.append({
                    'sentence_id': sentence_id,
                    'index': parts[0],
                    'word': parts[1],
                    'ner_tag': parts[2]
                })
    return pd.DataFrame(data)

train_df = load_data_to_dataframe('data/train')
dev_df = load_data_to_dataframe('data/dev')

def load_test_data_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as f:
        sentence_id = 0
        for line in f:
            if line.strip() == "":
                sentence_id += 1
            else:
                parts = line.strip().split()
                if len(parts) >= 2:  # Ensure the line has at least index and word
                    data.append({
                        'sentence_id': sentence_id,
                        'index': parts[0],
                        'word': parts[1]
                    })
    return pd.DataFrame(data)


# In[6]:


train_words = set(train_df['word'].unique())
special_tag = ['<PAD>', '<UNK>']
word_index= {}

##adding Padding and Unkown to the index
word_index = {token: idx for idx, token in enumerate(special_tag)}
word_index.update({word: idx + len(special_tag) for idx, word in enumerate(train_words)})


# In[7]:


#Extracting unique NER tags from the training dataset and storing them in a set
norm_tags = set(train_df['ner_tag'].unique())
tag_index = {tag: i for i, tag in enumerate(norm_tags)}
#Adding a special padding token at the end of the dictionary with a new index
tag_index['<PAD>'] = len(tag_index)


# In[8]:


print(f"Vocabulary size: {len(word_index)}")
print(f"Number of NER tags: {len(tag_index)}")


# In[9]:


# Mapping each word in dataframe to its corresponding index from the word_index dictionary
# Using '<UNK>' as a default index for words that are not found in the dictionary
train_df['word_idx'] = train_df['word'].map(lambda x: word_index.get(x, word_index['<UNK>']))
train_df['tag_idx'] = train_df['ner_tag'].map(tag_index)

dev_df['word_idx'] = dev_df['word'].map(lambda x: word_index.get(x, word_index['<UNK>']))
dev_df['tag_idx'] = dev_df['ner_tag'].map(tag_index)


# In[10]:


# Grouping indices by sentence ID in the dataframes and converting them into lists
train_sentences = train_df.groupby('sentence_id')['word_idx'].apply(list).tolist()
train_labels = train_df.groupby('sentence_id')['tag_idx'].apply(list).tolist()

dev_sentences = dev_df.groupby('sentence_id')['word_idx'].apply(list).tolist()
dev_labels = dev_df.groupby('sentence_id')['tag_idx'].apply(list).tolist()


# In[11]:


# Loading test data
test_df = load_test_data_to_dataframe('data/test')
test_df['word_idx'] = test_df['word'].map(lambda x: word_index.get(x, word_index['<UNK>']))
test_sentences = test_df.groupby('sentence_id')['word_idx'].apply(list).tolist()
test_sentence_dataset = list(zip(test_sentences, [None] * len(test_sentences)))  # No labels


# In[21]:


from torch.nn.utils.rnn import pad_sequence
#padding sequences of word and label indices to a uniform length
def pad_seq(batch):
    sentences, labels = zip(*batch)
    # Converting sentences to tensors and padding them with the predefined padding index
    sentences_padded = pad_sequence([torch.tensor(s) for s in sentences], batch_first=True, padding_value=word_index['<PAD>'])
    # Converting labels to tensors and padding them with the predefined padding index
    labels_padded = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=tag_index['<PAD>'])
    return sentences_padded, labels_padded
    
train_sentence_dataset = list(zip(train_sentences, train_labels))
dev_sentence_dataset = list(zip(dev_sentences, dev_labels))

train_loader = DataLoader(train_sentence_dataset, batch_size=32, shuffle=True, collate_fn=pad_seq)
dev_loader = DataLoader(dev_sentence_dataset, batch_size=32, shuffle=False, collate_fn=pad_seq)

test_loader = DataLoader(test_sentence_dataset, batch_size=32, shuffle=False, collate_fn=pad_seq)


# In[13]:


# HYPER-PARAMETERS
embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128
batch_size = 32
learning_rate = 0.1
num_epochs = 100
clip_value = 5
patience = 6
num_tags=len(tag_index)
vocab=len(word_index)


# In[14]:


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_output_dim, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        
        return logits


# In[15]:


##If a GPU is available, device will be set to "cuda"; otherwise, it will fall back to "cpu" ...trying to prevent memory error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##Moving the model to device
model = BiLSTM(vocab, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout)
model.to(device)


# In[16]:


Loss_Function = nn.CrossEntropyLoss(ignore_index=tag_index['<PAD>'])
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.5, verbose=True)

# Early stopping
early_stopping_counter = 0
best_f1_score = -1
patience = 6
clip_value = 5
num_epochs=50


# In[17]:


#training the BiLSTM model with a learning rate scheduler

def train_with_scheduler(model, train_loader, Loss_Function, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0

        for sentences, labels in train_loader:
            sentences = sentences.to(device)
            labels = labels.to(device)
             # Resetting gradients before backpropagation
            optimizer.zero_grad()

            predictions = model(sentences)
            # Reshaping predictions and labels for loss calculation
            predictions = predictions.view(-1, len(tag_index)) 
            labels = labels.view(-1)

            loss = Loss_Function(predictions, labels)
            loss.backward() # Performing backpropagation

            # Applying gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step() # Updating model parameters

            total_loss += loss.item() * sentences.size(0)
            total_samples += sentences.size(0)

        # Update learning rate
        scheduler.step(total_loss / total_samples)

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# In[18]:


#validating the BiLSTM model using loss, accuracy, precision, recall, and F1-score
def validate_with_metrics(model, dev_loader, Loss_Function, num_labels):
    model.eval()
    epoch_loss = 0
    total_accuracy = 0
    total_samples = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():# Disabling gradient calculations
        for sentences, labels in dev_loader:
            sentences = sentences.to(device)
            labels = labels.to(device)

            logits = model(sentences)
            logits = logits.view(-1, num_labels)# Reshaping logits for loss calculation
            labels = labels.view(-1)

            loss = Loss_Function(logits, labels)
            epoch_loss += loss.item()

            labels_cpu = labels.cpu().numpy()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels.extend(labels_cpu)
            predicted_labels.extend(predictions)

            # Creating a mask to exclude padding tokens from accuracy calculation
            mask = labels != tag_index['<PAD>']
            correct_predictions = (predictions[mask] == labels_cpu[mask]).sum()
            accuracy = correct_predictions / len(labels_cpu[mask])

            total_accuracy += accuracy
            total_samples += 1

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro', zero_division=0)

    avg_loss = epoch_loss / total_samples
    avg_accuracy = (total_accuracy / total_samples) * 100

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
    print(f"Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%")

    return avg_loss, avg_accuracy, precision, recall, f1


# In[19]:


def train_and_validate(model, train_loader, dev_loader, Loss_Function, optimizer, scheduler, num_epochs):
    best_f1_score = -1
    patience = 6

    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}")
        train_with_scheduler(model, train_loader, Loss_Function, optimizer, scheduler, num_epochs=1)

        print(f"Validating Epoch {epoch + 1}/{num_epochs}")
        avg_loss, avg_accuracy, precision, recall, f1 = validate_with_metrics(model, dev_loader, Loss_Function, num_labels=len(tag_index))

        if f1 > best_f1_score:
            best_f1_score = f1
            torch.save(model.state_dict(), "Blstm1.pt")


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support
train_and_validate(model, train_loader, dev_loader, Loss_Function, optimizer, scheduler, num_epochs)


# In[ ]:


def save_predictions_dev(model, data_loader, output_file, original_data, tag_index):
    model.eval()
    predictions = []

    with torch.no_grad():
        for sentences, _ in data_loader:
            sentences = sentences.to(device)

            # Forward pass
            logits = model(sentences)
            preds = torch.argmax(logits, dim=2)  # Get predicted tags

            # Append predictions
            predictions.extend(preds.cpu().numpy())

    # Creating a reverse mapping from tag index to tag label
    index_to_tag = {index: tag for tag, index in tag_index.items()}

    # Saving predictions to file ## we dont need to load test data earlier
    with open(output_file, "w") as f:
        for sentence_id, preds in enumerate(predictions):
            sentence_data = original_data[original_data['sentence_id'] == sentence_id]
            words = sentence_data['word'].tolist()
            indices = sentence_data['index'].tolist()

            for idx, word, pred_tag in zip(indices, words, preds):
                # Convert the predicted tag index back to the actual tag
                tag = index_to_tag[pred_tag]
                f.write(f"{idx} {word} {tag}\n")
            f.write("\n")  # Add a newline after each sentence
save_predictions_dev(model, dev_loader, "dev1.out", dev_df, tag_index)


# In[ ]:


def save_predictions_test(model, data_loader, output_file, original_data, tag_index):
    model.eval()
    predictions = []

    with torch.no_grad():
        for sentences, _ in data_loader:
            sentences = sentences.to(device)

            # Forward pass
            logits = model(sentences)
            preds = torch.argmax(logits, dim=2)  # Get predicted tags

            # Append predictions
            predictions.extend(preds.cpu().numpy())

    # Creating a reverse mapping from tag index to tag label
    index_to_tag = {index: tag for tag, index in tag_index.items()}

    # Saving predictions to file
    with open(output_file, "w") as f:
        for sentence_id, preds in enumerate(predictions):
            # Get the original words and indices for this sentence
            sentence_data = original_data[original_data['sentence_id'] == sentence_id]
            words = sentence_data['word'].tolist()
            indices = sentence_data['index'].tolist()

            # Write each word, index, and predicted tag to the file
            for idx, word, pred_tag in zip(indices, words, preds):
                # Converting the predicted tag index back to the actual tag
                tag = index_to_tag[pred_tag]
                f.write(f"{idx} {word} {tag}\n")
            f.write("\n") 
save_predictions_test(model, dev_loader, "test1.out", test_df, tag_index)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




