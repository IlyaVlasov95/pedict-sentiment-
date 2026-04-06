import pandas as pd
import numpy as np
import os
import json

from sklearn.preprocessing import (
    MinMaxScaler, 
    RobustScaler, 
    StandardScaler, 
    MaxAbsScaler, 
    Normalizer
)

import pandas as pd
import numpy as np
import os
import json
from scipy import stats

from sklearn.preprocessing import (
    MinMaxScaler, 
    RobustScaler, 
    StandardScaler, 
    MaxAbsScaler, 
    Normalizer,
    PowerTransformer
)


def normalize_data_with_scalers(df, scaler_config, is_file_path=False):
    """
    Нормализует данные с использованием различных скейлеров и трансформаций.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный DataFrame для нормализации
    scaler_config : dict или str
        Конфигурация скейлеров (словарь или путь к JSON-файлу)
    is_file_path : bool
        Флаг, указывающий, является ли scaler_config путем к файлу
        
    Returns:
    --------
    pd.DataFrame
        Нормализованный DataFrame

    example:
    --------
    config = {
    'Chain': {
        'vector_z': ['YeoJohnson', 'StandardScaler',{'Power':0.5}]  # Сначала нормализуем распределение, потом масштабируем
    },
    'BoxCox': ['vector_x', 'vector_y']  # Для уже нормальных распределений
    }

    """
    if is_file_path:
        if not os.path.exists(scaler_config):
            raise FileNotFoundError(f'Файл не найден: {scaler_config}')
        with open(scaler_config, 'r', encoding='utf-8') as f:
            scaler_columns_dict = json.load(f)
    else:
        scaler_columns_dict = scaler_config
    
    
    available_scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'StandardScaler': StandardScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'Normalizer': Normalizer(),
        'PowerTransformer_yj': PowerTransformer(method='yeo-johnson', standardize=True),
        'PowerTransformer_bc': PowerTransformer(method='box-cox', standardize=True),
        'Log': 'log',
        'Log1p': 'log1p',
        'Reciprocal': 'reciprocal',
        'Sqrt': 'sqrt',
        'Cbrt': 'cbrt',  # Cube root (корень 3-й степени)
        'BoxCox': 'boxcox',
        'YeoJohnson': 'yeojohnson',
        'Power': 'power'  # Для возведения в произвольную степень
    }
    
    result = pd.DataFrame(index=df.index)
    used_columns = []
    intermediate_data = df.copy()
    
    for scaler_name, columns in scaler_columns_dict.items():
        if scaler_name == 'Chain':
            # Специальная обработка для цепочек скейлеров
            for col, scaler_chain in columns.items():
                if not isinstance(scaler_chain, list):
                    scaler_chain = [scaler_chain]
                
                if col not in intermediate_data.columns:
                    print(f"Предупреждение: столбец '{col}' не найден в DataFrame")
                    continue
                    
                col_data = intermediate_data[[col]].copy()
                
                # Последовательно применяем каждый скейлер
                for sc_item in scaler_chain:
                    col_data = _apply_single_transform(col_data, sc_item, available_scalers, col)
                
                intermediate_data[col] = col_data
                used_columns.append(col)
                
        elif scaler_name == 'Power':
            # Специальная обработка для возведения в степень
            # Формат: {'Power': {'column_name': exponent, ...}}
            for col, exponent in columns.items():
                if col not in intermediate_data.columns:
                    print(f"Предупреждение: столбец '{col}' не найден в DataFrame")
                    continue
                    
                intermediate_data[col] = intermediate_data[col] ** exponent
                used_columns.append(col)
                
        else:
            # Обычная обработка для одиночных скейлеров
            if scaler_name not in available_scalers:
                raise ValueError(f"Скейлер {scaler_name} не поддерживается")
            if not columns:
                continue
            
            # Проверяем наличие столбцов
            missing_cols = [col for col in columns if col not in intermediate_data.columns]
            if missing_cols:
                print(f"Предупреждение: столбцы {missing_cols} не найдены в DataFrame")
                columns = [col for col in columns if col in intermediate_data.columns]
                if not columns:
                    continue
            
            # Применяем трансформацию
            intermediate_data = _apply_transform_to_columns(
                intermediate_data, columns, scaler_name, available_scalers
            )
            used_columns.extend(columns)
    
    # Сохраняем порядок столбцов как в исходном DataFrame
    result = intermediate_data.copy()
    result = result[df.columns.intersection(result.columns, sort=False)]
    
    return result


def _apply_single_transform(col_data, transform_spec, available_scalers, col_name):
    """
    Применяет одну трансформацию к данным столбца.
    
    Parameters:
    -----------
    col_data : pd.DataFrame
        Данные столбца (одномерный DataFrame)
    transform_spec : str, dict или tuple
        Спецификация трансформации
    available_scalers : dict
        Словарь доступных скейлеров
    col_name : str
        Имя столбца (для отладки)
        
    Returns:
    --------
    pd.DataFrame
        Трансформированные данные
    """
    # Если это словарь с параметрами (например, {'Power': 2})
    if isinstance(transform_spec, dict):
        scaler_name = list(transform_spec.keys())[0]
        param = transform_spec[scaler_name]
    else:
        scaler_name = transform_spec
        param = None
    
    if scaler_name not in available_scalers:
        raise ValueError(f"Скейлер {scaler_name} не поддерживается в цепочке")
    
    transform_type = available_scalers[scaler_name]
    
    # Применяем трансформацию
    if transform_type == 'log':
        return np.log(col_data)
    elif transform_type == 'log1p':
        return np.log1p(col_data)
    elif transform_type == 'reciprocal':
        return 1 / col_data
    elif transform_type == 'sqrt':
        return col_data ** 0.5
    elif transform_type == 'cbrt':
        return col_data ** (1/3)
    elif transform_type == 'boxcox':
        result, lambda_param = stats.boxcox(col_data[col_data.columns[0]])
        #print(f"BoxCox для '{col_name}': оптимальное λ = {lambda_param:.4f}")
        return pd.DataFrame(result, columns=col_data.columns, index=col_data.index)
    elif transform_type == 'yeojohnson':
        col_data_float = col_data.astype('float')
        result, lambda_param = stats.yeojohnson(col_data_float[col_data_float.columns[0]])
        #print(f"YeoJohnson для '{col_name}': оптимальное λ = {lambda_param:.4f}")
        return pd.DataFrame(result, columns=col_data.columns, index=col_data.index)
    elif transform_type == 'power':
        if param is None:
            raise ValueError(f"Для Power-трансформации необходимо указать степень")
        return col_data ** param
    elif isinstance(transform_type, str):
        raise ValueError(f"Неизвестный тип трансформации: {transform_type}")
    else:
        # Это sklearn scaler
        return pd.DataFrame(
            transform_type.fit_transform(col_data),
            columns=col_data.columns,
            index=col_data.index
        )


def _apply_transform_to_columns(df, columns, scaler_name, available_scalers):
    """
    Применяет трансформацию к списку столбцов.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными
    columns : list
        Список столбцов для трансформации
    scaler_name : str
        Имя скейлера
    available_scalers : dict
        Словарь доступных скейлеров
        
    Returns:
    --------
    pd.DataFrame
        DataFrame с трансформированными данными
    """
    transform_type = available_scalers[scaler_name]
    
    if transform_type == 'log':
        df[columns] = np.log(df[columns])
    elif transform_type == 'log1p':
        df[columns] = np.log1p(df[columns])
    elif transform_type == 'reciprocal':
        df[columns] = 1 / df[columns]
    elif transform_type == 'sqrt':
        df[columns] = df[columns] ** 0.5
    elif transform_type == 'cbrt':
        df[columns] = df[columns] ** (1/3)
    elif transform_type == 'boxcox':
        for col in columns:
            result, lambda_param = stats.boxcox(df[col])
            #print(f"BoxCox для '{col}': оптимальное λ = {lambda_param:.4f}")
            df[col] = result
    elif transform_type == 'yeojohnson':
        for col in columns:
            df[col] = df[col].astype('float')
            result, lambda_param = stats.yeojohnson(df[col])
            #print(f"YeoJohnson для '{col}': оптимальное λ = {lambda_param:.4f}")
            df[col] = result
    elif isinstance(transform_type, str):
        raise ValueError(f"Неизвестный тип трансформации: {transform_type}")
    else:
        # Это sklearn scaler
        scaled = transform_type.fit_transform(df[columns])
        df[columns] = pd.DataFrame(
            scaled,
            columns=columns,
            index=df.index
        )
    
    return df
