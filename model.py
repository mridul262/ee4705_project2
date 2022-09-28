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

def getBertModel():
    # Import the DistilBert pretrained model
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # freeze all the parameters. This will prevent updating of model weights during fine-tuning.
    for param in bert.parameters():
        param.requires_grad = False
    return bert
