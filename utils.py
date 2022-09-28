import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import optim
import random
import transformers
from transformers import AutoModel, BertTokenizerFast, DistilBertTokenizer, DistilBertModel
from torchinfo import summary
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from model import BERT_Arch, getBertModel

# All the intents/classes the nlp model will be classified into
df = pd.read_csv("./data/intents.csv")
movie_data = pd.read_csv('./data/movie_data.csv')
# df.head()

# Converting the labels into encodings
le = LabelEncoder()
df['Label_Encoded'] = le.fit_transform(df['Label'])
df.head()
# check class distribution
df['Label_Encoded'].value_counts(normalize = True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = BERT_Arch(getBertModel())
model.load_state_dict(torch.load('./trained_models/task_nlp_trained.pkl'))

# CONSTANTS #
MAX_SEQ_LEN = 9
KEYWORDS_TO_GENRE = {
    'scary': 'horror',
    'horror': 'horror',
}

BENCHMARK_YEAR = 2015
KEYWORDS_TO_TIME = {
    'now': '>',
    'present': '>',
    'new': '>',
    'latest': '>',
    'late': '>',
    'lately': '>',
    'young': '>',
    'recent': '>',
    'newest': '>',
    'past': '<',
    'old': '<',
    'oldest': '<',
    'early': '<',
    'earliest': '<',
}

BENCHMARK_RATING = 7.5

KEYWORDS_TO_RATING = {
    'good': '>',
    'great': '>',
    'excellent': '>',
    'atg': '>',
    'all time great': '>',
    'best': '>',
    'nice': '>',
    'top': '>',
    'bad': '<',
    'poor': '<',
    'worst': '<',
    'horrible': '<',
    'terrible': '<',
    'lousy': '<',
}

field_dictonary = {
  'Genre': KEYWORDS_TO_GENRE,
  'Time': KEYWORDS_TO_TIME,
  'Rating': KEYWORDS_TO_RATING
}

field_benchmark_dictionary = {
  'Time': BENCHMARK_YEAR,
  'Rating': BENCHMARK_RATING,
}

# Functions #

def get_prediction(str):
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    model.eval()

    tokens_test_data = tokenizer(
    test_text,
    max_length = MAX_SEQ_LEN,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    preds = model(test_seq, test_mask)
    preds = preds.detach().numpy()
    preds = np.argmax(preds, axis = 1)
    print('Intent Identified: ', le.inverse_transform(preds)[0])
    return le.inverse_transform(preds)[0]



def find_field_value(field, values, inputDataFrame):
    """Returns the movies that match a particular genre

    Args:
        field (str): Fieldname to filter on
        values (list[str]): List of values
        inputDataFrame (pandas.DataFrame): Dataframe containing the movie data

    Returns:
        pandas.DataFrame: A pandas dataframe filtered by the values input to the function
    """
    returnData = inputDataFrame
    for value in values:
       returnData = returnData[returnData[field].str.contains(value, regex=False)]
    return returnData

def find_relative_field(field, value, condition, inputDataFrame):
    """Returns the movies that match a particular genre

    Args:
        field (str): Fieldname to filter on
        value (str): the benchmark values to be compared against
        condition (str): the operator condition
        inputDataFrame (pandas.DataFrame): A pandas dataframe filtered by the values input to the function

    Returns:
        pandas.DataFrame: A pandas dataframe filtered by the values input to the function
    """
    returnData = inputDataFrame
    if (condition == '>'):
        returnData = returnData[(returnData[field] > value)]
    elif (condition == '<'):
        returnData = returnData[(returnData[field] < value)]
    elif (condition == '<='):
        returnData = returnData[(returnData[field] <= value)]
    elif (condition == '>='):
        returnData = returnData[(returnData[field] >= value)]
    elif (condition == '=='):
        returnData = returnData[(returnData[field] == value)]
    return returnData

def find_keywords_field(data_frame, message_arr, field, field_type):
  """Returns a dataframe containing the filtered movies based on the field, the type of the field and the input message

    Args:
        data_frame (pandas.DataFrame): Input Dataframe that needs to be filtered
        message_arr (str): the message based on which the DataFrame will be filtered
        field (str): field on which the DataFrame will be filtered
        field_type ('RELATIVE' | 'VALUE'): the type of the field

    Returns:
        pandas.DataFrame: the DataFame containing the filtered movie data
  """
  if field_type == 'VALUE':
    if field in field_dictonary:
      local_field_dict = field_dictonary[field]
      for i in message_arr:
        if i in local_field_dict:
          i = local_field_dict[i]
        if(find_field_value(field, [i.title()], data_frame).shape[0]) != 0:
          return find_field_value(field, [i.title()], data_frame)
    else:
      for i in message_arr:
        print(i)
        if(find_field_value(field, [i.title()], data_frame).shape[0]) != 0:
          print('inside if')
          return find_field_value(field, [i.title()], data_frame)
  elif field_type == 'RELATIVE':
    if field in field_dictonary:
      local_field_dict = field_dictonary[field]
      for i in message_arr:
        if i in local_field_dict:
          i = local_field_dict[i]
          print(i)
          if field in field_benchmark_dictionary:
            print(field)
            value = field_benchmark_dictionary[field]
            if(find_relative_field(field, value, i, data_frame).shape[0]) != 0:
              return find_relative_field(field, value, i, data_frame)
    else:
      for i in message_arr:
        print(i)
        if(find_relative_field(field, value, i, data_frame).shape[0]) != 0:
          print('inside if')
          return find_relative_field(field, value, i, data_frame)

def get_response(message, movie_data):
    """For a given message input a text response is returned

    Args:
        message (str): a str containing the message that needs to be responded to

    Returns:
        str: the response message
    """
    intent = get_prediction(message).lower()
    message_arr = message.split(' ')
    # intent_list = df['Label'].unique()
    response = 'Sorry, could you repeat that again'
    if intent == 'start':
      response = 'Hello, ask me for movie recommendations!'
    elif intent == 'question':
      response = 'What genre would you like?'
    elif intent == 'genre':
      movie_data = find_keywords_field(movie_data, message_arr, 'Genre', 'VALUE')
      response = 'how old should the movie be?'
    elif intent == 'time':
      movie_data = find_keywords_field(movie_data, message_arr, 'Year', 'RELATIVE')
      response = 'how rated should the movie be'
    elif intent == 'cast':
      movie_data = find_keywords_field(movie_data, message_arr, 'Actors', 'VALUE')
      response = 'We have shortlisted some movies for you!'
    elif intent == 'rating':
      movie_data = find_keywords_field(movie_data, message_arr, 'Rating', 'RELATIVE')
      response = 'whose movies would you like to watch'
    elif intent == 'end':
      response = 'Goodbye!'
      print(movie_data)
      print(response)
      return movie_data
    return response, movie_data


def get_response_text_input(): 
  movie_data = pd.read_csv('./data/movie_data.csv')
  movie_data.head()

  while True:
    message = input('Say something to start')
    intent = get_prediction(message).lower()
    message_arr = message.split(' ')
    # intent_list = df['Label'].unique()
    response = 'Sorry, could you repeat that again'
    if intent == 'start':
      response = 'Hello, ask me for movie recommendations!'
    elif intent == 'question':
      response = 'What genre would you like?'
    elif intent == 'genre':
      movie_data = find_keywords_field(movie_data, message_arr, 'Genre', 'VALUE')
      response = 'how old should the movie be?'
    elif intent == 'time':

      response = 'whose movies would you like to watch'
    elif intent == 'cast':
      movie_data = find_keywords_field(movie_data, message_arr, 'Actors', 'VALUE')
    elif intent == 'end':
      response = 'Goodbye!'
      print(movie_data)
      print(response)
      return movie_data
    print(response)

