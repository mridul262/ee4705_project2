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

CLASS_LABELS = le.classes_
CLASS_LABELS_ENCODED = le.transform(le.classes_)
CHATBOT_SEQUENCE = ['Start', 'Question', 'Genre', 'Time', 'Cast', 'Rating', 'End']

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

BENCHMARK_RATING = 7.49

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
    if type(returnData) == type(None):
      return returnData
    for value in values:
       returnData = returnData[returnData[field].str.contains(value + ',', regex=False)]
       if returnData.shape[0] == 0:
        returnData = inputDataFrame
        returnData = returnData[returnData[field].str.contains(value + ' ', regex=False)]
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
    if field == 'Time':
      field = 'Year'
    returnData = inputDataFrame
    if type(returnData) == type(None):
      return returnData
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
    if field == 'Year':
      field = 'Time'

    if field_type == 'VALUE':
        if field in field_dictonary:
            local_field_dict = field_dictonary[field]
            for i in message_arr:
                if i in local_field_dict:
                    i = local_field_dict[i]
                trimmedDF = find_field_value(field, [i.title()], data_frame)
                if type(trimmedDF) != type(None) and (trimmedDF.shape[0]) != 0:
                    return trimmedDF
        else:
            for i in message_arr:
                trimmedDF = find_field_value(field, [i.title()], data_frame)
                if type(trimmedDF) != type(None) and (trimmedDF.shape[0]) != 0:
                    return trimmedDF
    elif field_type == 'RELATIVE':
        if field in field_dictonary:
            local_field_dict = field_dictonary[field]
            for i in message_arr:
                if i in local_field_dict:
                    i = local_field_dict[i]
                    if field in field_benchmark_dictionary:
                        value = field_benchmark_dictionary[field]
                        trimmedDF = find_relative_field(field, value, i, data_frame)
                        if type(trimmedDF) != type(None) and (trimmedDF.shape[0]) != 0:
                            return trimmedDF
    else:
        for i in message_arr:
            trimmedDF = find_relative_field(field, value, i, data_frame)
            if type(trimmedDF) != type(None) and (trimmedDF.shape[0]) != 0:
                return trimmedDF

    return None

def get_film_title(movie_data):
  return_string = 'Here are some movies I\'d like to recommend: '
  if type(movie_data) != type(None) and movie_data.shape[0] > 0:
    total_rows = movie_data.shape[0]
    loop_range = min(total_rows, 3)
    # return a maximum of three movies
    for i in range(loop_range):
      return_string += movie_data.iloc[i]['Title']
      if i < (loop_range - 1):
        return_string += ', '
  else:
    return_string = 'Sorry, no movies matched your filter.'
  return return_string
      

def get_response(message, movie_data, intent_state):
    """For a given message input a text response is returned along with the filtered dataframe.
    Use this with the tts and stt functions

    Args:
        message (str): a str containing the message that needs to be responded to

    Returns:
        str: the response message
    """
    intent = get_prediction(message).lower()
    if intent != CHATBOT_SEQUENCE[intent_state].lower():
      response = 'Sorry, could you repeat that again'
    else:
      message_arr = message.split(' ')
      # intent_list = df['Label'].unique()
      # response = 'Sorry, could you repeat that again'
      if intent == 'start':
        response = 'Hello, ask me for movie recommendations!'
      elif intent == 'question':
        response = 'What genre would you like?'
      elif intent == 'genre':
        movie_data = find_keywords_field(movie_data, message_arr, 'Genre', 'VALUE')
        response = 'how old should the movie be?'
      elif intent == 'time':
        movie_data = find_keywords_field(movie_data, message_arr, 'Year', 'RELATIVE')
        response = "whose movie would you like to watch?"
      elif intent == 'cast':
        movie_data = find_keywords_field(movie_data, message_arr, 'Director', 'VALUE')
        response = 'how good should the movie be?'
      elif intent == 'rating':
        movie_data = find_keywords_field(movie_data, message_arr, 'Rating', 'RELATIVE')
        response = get_film_title(movie_data)
      elif intent == 'end':
        response = 'Goodbye'
      intent_state += 1
    return response, movie_data, intent_state


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