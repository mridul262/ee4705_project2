def find_field_value(field, values, inputDataFrame):
    """Returns the movies that match a particular genre

    Args:
        field (str): Fieldname to filter on
        values (list[str]): List of values

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
        values (list[str]): List of values

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

# find_field_value('Rating', ['James Gunn'])