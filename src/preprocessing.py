import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

import constants


def cache_file(f, extended_file_path):
    """
    Caches a file in the output folder. The filename should not contain the '.pickle' extension.
    """
    file = open(extended_file_path, "wb")
    pickle.dump(f, file)
    file.close()


def clean(filename):
    """Takes as input a filename of either timeseries or soil data and cleans data (drops fips that contain bad quality data, interpolates score, drops a useless column).
    Returns the cleaned data

    Args:
        filename (str): file absolute or relative path

    Returns:
        pd.DataFrame: dataframe containing the clean data
    """
    # Read the data
    data = pd.read_csv(filename)

    # Fips with bad quality data or sparse data (found in first data exploration)
    fips_to_drop = constants.FIPS_TO_DROP

    # Dropping bad fips
    data = data[~data['fips'].isin(fips_to_drop)]

    # Derive from the shape of the data if it is soil data or timeseries data
    nrwows, ncols = data.shape
    if ncols == 21:
        print('The dataframe you inputted contains timeseries data, dropping fips that contain bad data, and interpolating score')
        # Interpolation
        data['score'] = (data[['fips', 'score']].groupby(['fips']).apply(lambda group: group.interpolate(method = "linear", limit_direction = "both")))['score']
    elif ncols == 32:
        print('The dataframe you inputted contains soil data, dropping fips that contain bad data, and dropping "CULT_LAND"')
        # Drop useless CULT_LAND variable in the soil data
        data.drop(columns = ['CULT_LAND'], inplace = True)
    else:
        print("There is an error in the dimensions of the file you want to load, check that you loaded soil data or timeseries")
    return data



def clean_and_combine(timeseries_filename, soil_filename):
    """Cleans and merges timeseries and soil_data

    Args:
        timeseries_filename (str): timeseries absolute or relative path
        soil_filename (_type_): soil data absolute or relative path

    Returns:
        pd.DataFrame: dataframe containing the cleaned and merged data
    """

    # CLean timeseries and soil data
    timeseries = clean(timeseries_filename)
    soil_data = clean(soil_filename)

    # Generate the output data
    output_data = pd.merge(timeseries, soil_data, on = 'fips', how = 'left')

    return output_data


def load_clean_data(data_name):
    """Loads clean data if exists, otherwise creates clean data and puts it in cache.

    Args:
        data_name (str): "soil", "train", "validation" or "test"
    """
    cleaned_file_name = {"soil": "cleaned_soil_data.pickle", "train": "cleaned_train_timeseries.pickle", 
                         "validation": "cleaned_validation_timeseries.pickle", 
                         "test": "cleaned_test_timeseries.pickle"}[data_name]
    cleaned_file_path = os.path.join(constants.DATA_PATH, cleaned_file_name)
    if os.path.exists(cleaned_file_path):
        print(f"Loading existing clean {data_name} data")
        return pickle.load(open(cleaned_file_path, "rb"))
    else:
        print(f"Clean {data_name} data could not be found, cleaning data and caching it...")
        filename = {"soil": "soil_data.csv", "train": "train_timeseries.csv", 
                    "validation": "validation_timeseries.csv", "test": "test_timeseries.csv"}[data_name]
        file_path = os.path.join(constants.DATA_PATH, filename)
        cleaned_df = clean(file_path)
        cache_file(cleaned_df, cleaned_file_path)


def cyclic_date(date):
    """Encodes string date into trigonometric coordinates. Two dates one year apart from each other have the same encoding.

    Args:
        date (str): date string

    Returns:
        Tuple(float, float): tuple of date coordinates
    """
    date = datetime.strptime(date, "%Y-%m-%d")
    day_number = date.timetuple().tm_yday
    return(np.sin(2 * np.pi * day_number / 366), np.cos(2 * np.pi * day_number / 366))


def transform_data_dumb(timeseries_data, 
    time_window = 2,
    target_size = 6):
    """Creates an array for regression, where X contains only the previous and the current days' drought scores
    This data will be used with predictors that do not involve statistical learning,
    only deduction of the output given observed drought values
    """
    X = np.empty((timeseries_data.shape[0]//time_window, time_window))
    y = np.empty((timeseries_data.shape[0]//time_window, target_size))
    count = 0
    list_of_fips = pd.unique(timeseries_data['fips']).tolist()
    minimum_start_index = 0
    for fips in tqdm(list_of_fips):
        restrained_timeseries = timeseries_data[timeseries_data['fips'] == fips].reset_index(drop = True)
        i = random.randint(minimum_start_index, minimum_start_index + time_window)
        while i + (time_window - 1) + target_size * 7 < restrained_timeseries.shape[0]:
            X[count, :] = restrained_timeseries['score'].iloc[i:i+time_window]
            y[count] = restrained_timeseries['score'][range(i + time_window - 1 + 7, i + time_window - 1 + 7*6 + 1, 7)]
            i += time_window
            count += 1
    X = X[:count]
    y = y[:count]
    return X, y
    

def transform_data_2d(timeseries_data, soil_data, use_lat_lon = True, time_window = 180, 
    target_size = 6, use_previous_year_data = False, use_previous_drought_scores = True, 
    training_set = True, step = 20):
    """From timeseries and soil data, generates a dataframe ready for use of ML techniques
    The variables in this dataframe are:
        - the encoded dates
        - the timeseries variables for the past time_window days
        - static soil data
    The starting day could is random to generate diverse training data.

    Args:
        timeseries_data (pd.DataFrame): timeseries df
        soil_data (pd.DataFrame): soil data df
        use_lat_lon (bool, optional): True if we want to include latitude and longitude. Defaults to True.
        time_window (int, optional): One training episode length. Defaults to 180.
        target_size (int, optional): Number of lead times we want to predict from now. Defaults to 6.
        use_previous_year_data (bool, optional): True if we want to include past year's data. Defaults to False.
        use_previous_drought_scores (bool, optional): True if we want to include past droughts scores. Defaults to True.
        training_set (bool, optional): True if we want to generate a training set. Defaults to True.
        step (int, optional): Validation and training sets interval between two episodes start dates. Defaults to 20.

    Returns:
        Tuple(np.ndarray[n_points, n_variables], np.ndarray[n_points, n_targets]): independent and dependent variables
    """
    weather_variables = timeseries_data.columns.drop(['fips', 'date', 'score'])
    n_weather_variables = len(weather_variables)
    n_time_variables = n_weather_variables + 2 #the '+2' here represents the encoded date
    main_variables = weather_variables
    if use_previous_drought_scores:
        weather_and_score_variables = timeseries_data.columns.drop(['fips', 'date'])
        n_weather_and_score_variables = len(weather_and_score_variables)
        n_time_variables = n_weather_and_score_variables + 2 #the '+2' here represents the encoded date
        main_variables = weather_and_score_variables
    if use_lat_lon:
        soil_variables = soil_data.columns.drop(['fips'])
        n_soil_variables = len(soil_variables)
    else:
        soil_variables = soil_data.columns.drop(['fips', 'lat', 'lon'])
        n_soil_variables = len(soil_variables)
    if use_previous_year_data:
        n_years = 2
    else:
        n_years = 1
    tot_variables = n_years * (time_window * n_time_variables) + n_soil_variables
    X = np.empty((timeseries_data.shape[0]//time_window, tot_variables))
    y = np.empty((timeseries_data.shape[0]//time_window, target_size))
    count = 0
    list_of_fips = pd.unique(timeseries_data['fips']).tolist()
    if use_previous_year_data:
        minimum_start_index = 365
    else:
        minimum_start_index = 0
    for fips in tqdm(list_of_fips):
        restrained_timeseries = timeseries_data[timeseries_data['fips'] == fips].reset_index(drop = True)
        i = random.randint(minimum_start_index, minimum_start_index + time_window)
        while i + (time_window - 1) + target_size * 7 < restrained_timeseries.shape[0]:
            X[count, : time_window * n_time_variables] = list(np.array([cyclic_date(restrained_timeseries['date'].iloc[j]) for j in range(i, i+time_window)]).flat) + restrained_timeseries[main_variables].iloc[i:i+time_window].to_numpy().flatten().tolist()
            if use_previous_year_data:
                X[count, time_window * n_time_variables : n_years * (time_window * n_time_variables)] = list(np.array([cyclic_date(restrained_timeseries['date'].iloc[j]) for j in range(i-365, i-365+time_window)]).flat) + restrained_timeseries[main_variables].iloc[i-365:i-365+time_window].to_numpy().flatten()
            X[count, - n_soil_variables:] = soil_data[soil_data['fips'] == fips][soil_variables].to_numpy().flatten()
            y[count] = restrained_timeseries['score'][range(i + time_window - 1 + 7, i + time_window - 1 + 7*6 + 1, 7)]
            if training_set:
                i += time_window # We skip all observations that were included in the construction of the current row
                                 # in order to avoid introducing correlation in the training set
            else:
                i += step
            count += 1
    X = X[:count]
    y = y[:count]
    return X, y


def transform_data_3d(timeseries_data, soil_data, use_lat_lon = True, time_window = 180, 
    target_size = 6, use_previous_year_data = False, use_previous_drought_scores = True, 
    training_set = True, step = 20):
    """From timeseries and soil data, generates tensors ready for use of Deep Learning techniques
    The variables in this dataframe are:
        - the encoded dates
        - the timeseries variables for the past time_window days
        - static soil data
    The starting day could is random to generate diverse training data.

    Args:
        timeseries_data (pd.DataFrame): timeseries df
        soil_data (pd.DataFrame): soil data df
        use_lat_lon (bool, optional): True if we want to include latitude and longitude. Defaults to True.
        time_window (int, optional): One training episode length. Defaults to 180.
        target_size (int, optional): Number of lead times we want to predict from now. Defaults to 6.
        use_previous_year_data (bool, optional): True if we want to include past year's data. Defaults to False.
        use_previous_drought_scores (bool, optional): True if we want to include past droughts scores. Defaults to True.
        training_set (bool, optional): True if we want to generate a training set. Defaults to True.
        step (int, optional): Validation and training sets interval between two episodes start dates. Defaults to 20.

    Returns:
        Tuple(np.ndarray[n_episodes, time_window, n_time_variables], np.ndarray[n_soil_variables], np.ndarray[n_episodes, target_size])
        : time variables tensor, static independent variables, and dependent variables
    """
    weather_variables = timeseries_data.columns.drop(['fips', 'date', 'score'])
    n_weather_variables = len(weather_variables)
    n_time_variables = n_weather_variables + 2 #the '+2' here represents the encoded date
    if use_previous_drought_scores:
        weather_and_score_variables = timeseries_data.columns.drop(['fips', 'date'])
        n_weather_and_score_variables = len(weather_and_score_variables)
        n_time_variables = n_weather_and_score_variables + 2 #the '+2' here represents the encoded date
    if use_lat_lon:
        soil_variables = soil_data.columns.drop(['fips'])
        n_soil_variables = len(soil_variables)
    else:
        soil_variables = soil_data.columns.drop(['fips', 'lat', 'lon'])
        n_soil_variables = len(soil_variables)
    if use_previous_year_data:
        n_years = 2
    else:
        n_years = 1
    X_time = np.empty((timeseries_data.shape[0]//time_window, time_window, n_time_variables))
    X_static = np.empty((timeseries_data.shape[0] // time_window, n_soil_variables))
    y_target = np.empty((timeseries_data.shape[0]//time_window, target_size))
    count = 0
    list_of_fips = pd.unique(timeseries_data['fips']).tolist()
    if use_previous_year_data:
        minimum_start_index = 365
    else:
        minimum_start_index = 0
    for fips in tqdm(list_of_fips):
        restrained_timeseries = timeseries_data[timeseries_data['fips'] == fips].reset_index(drop = True).copy(deep=True)
        X_s = soil_data[soil_data["fips"] == fips][soil_variables].values[0]
        i = random.randint(minimum_start_index, minimum_start_index + time_window)
        while i + (time_window - 1) + target_size * 7 < restrained_timeseries.shape[0]:
            X_time[count, :, : n_time_variables-2] = restrained_timeseries[i : i + time_window][weather_and_score_variables]
            X_time[count, :, n_time_variables-2:] = np.array([cyclic_date(restrained_timeseries['date'].iloc[j]) for j in range(i, i+time_window)])
            X_static[count] = X_s
            y_target[count] = restrained_timeseries['score'][range(i + time_window - 1 + 7, i + time_window - 1 + 7*6 + 1, 7)]
            if training_set:
                i += time_window # We skip all observations that were included in the construction of the current row
                                 # in order to avoid introducing correlation in the training set
            else:
                i += step
            count += 1
    X_time = X_time[:count]
    X_static = X_static[:count]
    y_target = y_target[:count]
    return [X_time, X_static, y_target]


def custom_scaler(X_train, X_valid = None, X_test = None):
    """ Uses the train data to figure out the transformations to be done. Applies this transformation to the validation and test data if available.

    Args:
        X_train (np.ndarray): training set
        X_valid (np.ndarray, optional): validation set. Defaults to None.
        X_test (np.ndarray, optional): test set. Defaults to None.

    Returns:
        the same number of arrays that were passed as arguments
    """
    transformer = RobustScaler().fit(X_train)
    X_train = transformer.transform(X_train)
    if X_valid is not None and X_test is not None:
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)
        return X_train, X_valid, X_test
    elif X_valid is not None:
        X_valid = transformer.transform(X_valid)
        return X_train, X_valid
    elif X_test is not None:
        X_test = transformer.transform(X_test)
        return X_train, X_test
    else:
        return X_train
    

def round_and_intify(y):
    """Rounds float scores to integers

    Args:
        y (np.ndarray): float scores

    Returns:
        np.ndarray: int scores
    """
    return np.clip(np.squeeze(y).round().astype('int'), 0, 5)


def get_dumb_data(data_name):
    filenames = {"train": ("X_train_dumb.pickle", "y_target_train_dumb.pickle"), 
                "validation": ("X_valid_dumb.pickle", "y_target_valid_dumb.pickle")}[data_name]
    file_paths = [os.path.join(constants.DATA_PATH, filename) for filename in filenames]

    if os.path.exists(file_paths[0]) and os.path.exists(file_paths[1]):
            return pickle.load(open(file_paths[0], "rb")), pickle.load(open(file_paths[1], "rb"))
            
    timeseries = load_clean_data(data_name)
    X_dumb, y_target_dumb = transform_data_dumb(timeseries)
    cache_file(X_dumb, file_paths[0])
    cache_file(y_target_dumb, file_paths[1])
    return X_dumb, y_target_dumb


def get_lower_dimension_1(data_name):
    filename = {"train": "train_timeseries_lower_dim_1.pickle", 
                "validation": "validation_timeseries_lower_dim_1.pickle", 
                "test": "test_timeseries_lower_dim_1.pickle"}[data_name]
    file_path = os.path.join(constants.DATA_PATH, filename)
    if os.path.exists(file_path):
        return pickle.load(open(file_path, "rb"))
    low_dim_1_ts = load_clean_data(data_name)
    low_dim_1_ts['T_COMP'] = low_dim_1_ts[['T2MWET', 'T2MDEW', 'T2M_MIN', 'T2M', 'QV2M', 'TS', 'T2M_MAX']].progress_apply(lambda x: np.mean(x), axis = 1)
    low_dim_1_ts['W_COMP'] = low_dim_1_ts[['WS10M', 'WS50M', 'WS10M_MIN', 'WS50M_MIN', 'WS10M_MAX', 'WS50M_MAX']].progress_apply(lambda x: np.mean(x), axis = 1)
    low_dim_1_ts.drop(columns = ['T2MWET', 'T2MDEW', 'T2M_MIN', 'T2M', 'QV2M', 'TS', 'T2M_MAX'], inplace = True)
    low_dim_1_ts.drop(columns = ['WS10M', 'WS50M', 'WS10M_MIN', 'WS50M_MIN', 'WS10M_MAX', 'WS50M_MAX'], inplace = True)
    cache_file(low_dim_1_ts, file_path)
    return low_dim_1_ts


def get_lower_dimension_2(data_name):
    filename = {"train": "train_timeseries_lower_dim_2.pickle", 
                "validation": "validation_timeseries_lower_dim_2.pickle", 
                "test": "test_timeseries_lower_dim_2.pickle"}[data_name]
    file_path = os.path.join(constants.DATA_PATH, filename)
    if os.path.exists(file_path):
        return pickle.load(open(file_path, "rb"))
    low_dim_2_ts = get_lower_dimension_1(data_name)
    low_dim_2_ts['W_RANGE_COMP'] = low_dim_2_ts[['WS10M_RANGE', 'WS50M_RANGE']].progress_apply(lambda x: np.mean(x), axis = 1)
    low_dim_2_ts.drop(columns = ['WS10M_RANGE', 'WS50M_RANGE'], inplace = True)
    cache_file(low_dim_2_ts, file_path)
    return low_dim_2_ts


def get_lower_dimension_3(data_name):
    filename = {"train": "train_timeseries_lower_dim_3.pickle", 
                "validation": "validation_timeseries_lower_dim_3.pickle", 
                "test": "test_timeseries_lower_dim_3.pickle"}[data_name]
    file_path = os.path.join(constants.DATA_PATH, filename)
    if os.path.exists(file_path):
        return pickle.load(open(file_path, "rb"))
    low_dim_3_ts = get_lower_dimension_2(data_name)
    low_dim_3_ts.drop(columns = ['W_RANGE_COMP'], inplace = True)
    cache_file(low_dim_3_ts, file_path)
    return low_dim_3_ts


def normalize(X_static, X_time, scaler_dict=None, scaler_dict_static=None, fit=False):
    if not scaler_dict:
        scaler_dict = {}
    if not scaler_dict_static:
        scaler_dict_static = {}
    for index in tqdm(range(X_time.shape[-1])):
        if fit:
            scaler_dict[index] = RobustScaler().fit(X_time[:, :, index].reshape(-1, 1))
        X_time[:, :, index] = (
            scaler_dict[index]
            .transform(X_time[:, :, index].reshape(-1, 1))
            .reshape(-1, X_time.shape[-2])
        )
    for index in tqdm(range(X_static.shape[-1])):
        if fit:
            scaler_dict_static[index] = RobustScaler().fit(
                X_static[:, index].reshape(-1, 1)
            )
        X_static[:, index] = (
            scaler_dict_static[index]
            .transform(X_static[:, index].reshape(-1, 1))
            .reshape(1, -1)
        )
    if fit:
        return X_time, X_static, scaler_dict, scaler_dict_static
    return X_time, X_static
