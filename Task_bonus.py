#!/usr/bin/env python
# coding: utf-8

# <center><h1>NLP_HomeWork4_BonusTask</h1></center>
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


# In[2]:


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


# In[3]:


# Function to Create Mappings Considering upper Case Letters and Individual  CREATE MAPPINGS Letters[for CNN]
def cnn_vocab_mappings(raw_data, unique_tags, threshold):
    word_freqs = Counter(word.lower() for words, _ in raw_data for word in words)
    filtered_words = [word.lower() for word, count in word_freqs.items() if count >= threshold]
    
    # print(filtered_words)
    word_index = {word: idx + 4 for idx, word in enumerate(filtered_words)}
    word_index['<pad>'] = 0
    word_index['<s>'] = 1
    word_index['</s>'] = 2
    word_index['<unk>'] = 3

    tag_index = {tag: idx + 3 for idx, tag in enumerate(unique_tags)}
    tag_index['<pad>'] = 0
    tag_index['<s>'] = 1
    tag_index['</s>'] = 2

    all_chars = {char for words, _ in raw_data for word in words for char in word}
    char_index = {char: idx + 2 for idx, char in enumerate(all_chars)}
    char_index['<pad>'] = 0
    char_index['<unk>'] = 1

    return word_index, tag_index, char_index

def pad_word_chars(chars, max_word_len, pad_idx):
    return chars + [pad_idx] * (max_word_len - len(chars))

def pad_sequences(batch, word_index, tag_index, char_index, pad_token='<pad>', init='<s>', eos='</s>', unk='<unk>'):
    max_len = max([len(seq) + 2 for seq, _ in batch])
    max_word_len = max([len(word) for words, _ in batch for word in words])

    padded_word_seqs = []
    padded_upper_seqs = []
    padded_char_seqs = []
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

        padded_chars = [[char_index.get(char, char_index['<unk>']) for char in word] for word in words]
        padded_chars = [pad_word_chars(chars, max_word_len, char_index[pad_token]) for chars in padded_chars]
        padded_chars.insert(0, [char_index[pad_token]] * max_word_len)
        padded_chars.append([char_index[pad_token]] * max_word_len)
        padded_chars += [[char_index[pad_token]] * max_word_len] * (max_len - len(padded_chars))
        padded_char_seqs.append(padded_chars)

    return torch.tensor(padded_word_seqs), torch.tensor(padded_upper_seqs), torch.tensor(padded_char_seqs), torch.tensor(padded_tag_seqs)

def preprocess(text, word_index, char_index, pad_token='<pad>', init='<s>', eos='</s>', unk='<unk>'):
    tokens = text.split()

    lower_tokens = text.lower().split()
    padded_tokens = [init] + lower_tokens + [eos]
    indices = [word_index.get(word, word_index[unk]) for word in padded_tokens]
    
    upper_indices = [0] + [int(token[0].isupper()) for token in tokens] + [0]

    char_indices = [[char_index.get(char, char_index[unk]) for char in word] for word in tokens]
    max_word_len = max([len(word_chars) for word_chars in char_indices]) + 2
    char_indices = [[char_index[pad_token]] * max_word_len] + char_indices + [[char_index[pad_token]] * max_word_len]
    char_indices_padded = [word_chars + [char_index[pad_token]] * (max_word_len - len(word_chars)) for word_chars in char_indices]

    return indices, upper_indices, char_indices_padded

# FUNCTION TO PREDICT RESULTS
def predict_tags(model, input_text, word_index, char_index, tag_index):
    model.eval()
    tokenized_input, upper_input, char_input = preprocess(input_text, word_index, char_index)
    input_tensor = torch.tensor([tokenized_input]).to(device)
    upper_tensor = torch.tensor([upper_input]).to(device)
    char_input_tensor = torch.tensor([char_input]).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor, upper_tensor, char_input_tensor)
    
    predicted_indices = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    predicted_tags = [tag_index[idx] for idx in predicted_indices][1:-1]

    return predicted_tags


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


# Everything same as Task 2 just Adding Char_index for CNN
train_file = "data/train" 
raw_data, unique_words, unique_tags = load_data_to_dataframe(train_file)
tokenized_data = [([word for word in words], [tag for tag in tags]) for words, tags in raw_data]
train_dataset = CustomDataset(tokenized_data)

dev_file = "data/dev" 
raw_data, unique_words, unique_tags = load_data_to_dataframe(dev_file)
tokenized_data = [([word for word in words], [tag for tag in tags]) for words, tags in raw_data]
dev_dataset = CustomDataset(tokenized_data)

word_index, tag_index, char_index = cnn_vocab_mappings(raw_data, unique_tags, threshold=1)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    collate_fn=lambda batch: pad_sequences(batch, word_index, tag_index, char_index),
    shuffle=True,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=8,
    collate_fn=lambda batch: pad_sequences(batch, word_index, tag_index, char_index),
    shuffle=True,
)


# In[6]:


# HYPER-PARAMETERS
vocab_size = len(word_index)
num_tags = len(tag_index)
embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128

# HYPER-PARAMETERS
vocab_size = len(word_index)
char_vocab_size = len(char_index)
num_tags = len(tag_index)

char_embedding_dim = 30
embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128


# In[7]:


##Same as Task 2 along with Char_inputs for CNN
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
            word_seqs, upper_seqs, char_inputs, tag_seqs = batch
            word_seqs = word_seqs.to(device)
            upper_seqs = upper_seqs.to(device)
            char_inputs = char_inputs.to(device)
            tag_seqs = tag_seqs.to(device)
            # Pass char_inputs to the model
            logits = model(word_seqs, upper_seqs, char_inputs)
            logits = logits.view(-1, num_tags)
            tag_seqs = tag_seqs.view(-1)

            loss = loss_function(logits, tag_seqs)
            total_loss += loss.item()

            labels = tag_seqs.cpu().numpy()
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels)

            _, pred_tags = torch.max(logits, 1)
            y_pred.extend(pred_tags.cpu().numpy())

            mask = labels != 0
            correct_predictions = (predicted_labels[mask] == labels[mask]).sum()
            accuracy = correct_predictions / len(labels[mask])
            
            total_accuracy += accuracy
            epoch_loss += loss
            total_amount += 1

    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    print(f"Validation Loss: {(epoch_loss/total_amount)}, Accuracy: {(total_accuracy/total_amount)*100}%")
    print(f"Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1_score * 100:.2f}%")
    return (epoch_loss/total_amount), (total_accuracy/total_amount)*100, precision*100, recall*100, f1_score*100


# In[8]:


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


class BiLSTM_CNN(nn.Module):
    def __init__(self, embedding_matrix, char_vocab_size, num_tags, char_embedding_dim, embedding_dim, hidden_dim, num_layers, dropout, linear_output_dim):
        super(BiLSTM_CNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False).to(torch.float32)
        self.upper_embedding = nn.Embedding(2, embedding_dim)
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_cnn = nn.Conv1d(char_embedding_dim, embedding_dim, kernel_size=3)
        
        self.lstm = nn.LSTM(embedding_dim * 3, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_output_dim, num_tags)

    def forward(self, x, upper_x, chars):
        x = self.embedding(x)
        upper_x = self.upper_embedding(upper_x)
        
        chars = self.char_embedding(chars)
        batch_size, max_seq_len, max_word_len, _ = chars.shape
        chars = chars.view(batch_size * max_seq_len, max_word_len, -1).permute(0, 2, 1)

        char_features = self.char_cnn(chars)
        char_features = nn.functional.relu(char_features)
        char_features, _ = torch.max(char_features, dim=-1)
        char_features = char_features.view(batch_size, max_seq_len, -1)
        # print(char_features)
        
        x = torch.cat([x, upper_x, char_features], dim=-1)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        logits = self.linear2(x)

        return logits
        


# In[10]:


# Training and Predictions and saving the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_model = None
highest_f1_score = 0

model = BiLSTM_CNN(embedding_matrix, char_vocab_size, num_tags, char_embedding_dim, embedding_dim, hidden_dim, num_layers, dropout, linear_output_dim)
model.to(device)

num_epochs = 25

loss_function = CrossEntropyLoss(ignore_index=tag_index['<pad>'])
optimizer = optim.SGD(model.parameters(), lr=0.25, momentum=0.9, weight_decay=0.00005)

patience = 5
writer = SummaryWriter()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, verbose=True)

early_stopping_counter = 0
best_f1_score = -1
clip_value = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in train_loader:
        inputs, upper_inputs, char_inputs, labels = batch

        optimizer.zero_grad()

        logits = model(inputs, upper_inputs, char_inputs)

        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        loss = loss_function(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        total_loss += loss.item() * 16
        total_samples += 16

    avg_train_loss = total_loss / total_samples
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    
    val_loss, val_accuracy, val_precision, val_recall, val_f1_score = validate_with_metrics(model, dev_loader, loss_function, num_tags)
torch.save(model.state_dict(), "Blstm_bonus.pt")
writer.close()


# In[11]:


#FUNCTION TO CREATE OUTPUT FILES
def save_predictions_dev(model, text_file, output_file, tag_to_index, word_to_index, char_to_index):
    with open(text_file, 'r') as input_file, open(output_file, 'w') as output_file:
        indices = []
        words = []
        tags = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0 and len(tags) > 0:
                    idx_to_tag = {idx: tag for tag, idx in tag_to_index.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags(model, new_text, word_to_index, char_to_index, idx_to_tag)

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

def save_predictions_test(model, text_file, output_file, tag_to_index, word_to_index, char_to_index):
    with open(text_file, 'r') as input_file, open(output_file, 'w') as output_file:
        indexs = []
        words = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0:
                    idx2tag = {idx: tag for tag, idx in tag_to_index.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags(model, new_text, word_to_index, char_to_index, idx2tag)

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


# In[12]:


# CREATING OUTPUT FILES
save_predictions_dev(model, "data/dev", "dev_bonus.out", tag_index, word_index,char_index)

save_predictions_test(model, "data/test", "test_bonus.out", tag_index, word_index,char_index)

model.eval()


# In[ ]:




