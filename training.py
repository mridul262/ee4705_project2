"""
File for training the model based on the training data and intents supplied
"""
# Pytorch related imports #
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import optim

# Transformers related imports for BERT #
import transformers
from transformers import AutoModel, BertTokenizerFast, DistilBertTokenizer, DistilBertModel

# Support modules #
import numpy as np
import pandas as pd
import re
import random
from torchinfo import summary
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import df, test_df

class BERT_Arch(nn.Module):
   """
    BERT Model Training Architecture
   """
   def __init__(self, bert):      
       super(BERT_Arch, self).__init__()
       # Using BERT on its own
       self.bert = bert 

       # Additional Training Layers #
       # dropout layer
       self.dropout = nn.Dropout(0.2)
      
       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,7)
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass
   def forward(self, sent_id, mask):
      #pass the inputs to the model  
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
      
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)
   
      # apply softmax activation
      x = self.softmax(x)
      return x

# In this example we have used all the utterances for training purpose
train_text, train_labels = df['Text'], df['Label_Encoded']
test_text, test_labels = test_df['Text'], test_df['Label_Encoded']
# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins = 10)

max_seq_len = 9

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

tokens_test = tokenizer(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

#define a batch size
batch_size = 16
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)
# sampler for sampling the data during testing
test_sampler = RandomSampler(test_data)
# DataLoader for test set
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

train_size = df.shape[0]
test_size = test_df.shape[0]

# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
      param.requires_grad = False
model = BERT_Arch(bert)
# push the model to GPU
# model = model.to(device)
summary(model)

# define the optimizer
optimizer = optim.AdamW(model.parameters(), lr = 1e-3)

#compute the class weights
class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
# print(class_wts)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
# loss function
cross_entropy = nn.NLLLoss(weight=weights) 
# We can also use learning rate scheduler to achieve better results
lr_sch = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# function to train the model
def train():
  
    model.train()
    total_loss = 0
    train_acc = 0
    # empty list to save model predictions
    total_preds=[]

    # iterate over batches
    for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
        # if step % 50 == 0 and not step == 0:
        # print(f'Batch {step}  of  {len(train_dataloader)}.')
        # push the batch to gpu
        # batch = [r.to(device) for r in batch] 
        sent_id, mask, labels = batch
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values
        # print(preds, labels)
        loss = cross_entropy(preds, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the    exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # clear calculated gradients
        optimizer.zero_grad()

        # We are not using learning rate scheduler as of now
        # lr_sch.step()
        # model predictions are stored on GPU. So, push it to CPU
        # preds=preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds.detach().numpy())
        # train accuracy
        _, train_predicted = torch.max(preds.data, 1)
        train_acc += (train_predicted == labels).sum().item()
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    train_acc /= train_size
    
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    #returns the loss and predictions
    return avg_loss, total_preds, train_acc

# empty lists to store training and validation loss of each epoch
train_losses=[]
train_acc_arr = []
test_acc_arr = []
best_accuracy = 0
# number of training epochs
epochs = 200
print('train_losses len =', len(train_losses))
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _, train_acc = train()
    
    # append training and validation loss
    train_losses.append(train_loss)
    train_acc_arr.append(train_acc)
    # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f'\nTraining Loss: {train_loss:.3f}')
    # ----------test----------
    model.eval()
    test_acc = 0.0
    for step, batch in test_dataloader:
        sent_id, mask, labels = batch
        test_output = model(sent_id, mask)
        _, predicted = torch.max(test_output.data, 1)
        test_acc += (predicted == labels).sum().item()
    test_acc /= test_size
    test_acc_arr.append(test_acc)
    if test_acc >= best_accuracy:
        torch.save(model.state_dict(), './trained_models/Task3_CNN_model.pkl')
        best_accuracy = test_acc

# torch.save(model.state_dict(), './trained_models/task_nlp_trained.pkl')

plt.plot([epoch for epoch in range(epochs)], train_losses)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.savefig('loss_entropy.png')
# print(train_losses)
