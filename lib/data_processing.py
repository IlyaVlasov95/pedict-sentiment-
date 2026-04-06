# lib/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import category_encoders as ce

def encode_data(df, encoder_columns_dict, y=None):
    """
    Универсальная функция для кодирования категориальных признаков с использованием различных методов.
    
    Args:
        df (pd.DataFrame): Исходный DataFrame с данными для кодирования
        encoder_columns_dict (dict): Словарь, где ключи - названия кодировщиков, 
                                   значения - списки колонок для применения каждого кодировщика
        y (pd.Series, optional): Целевая переменная, требуется для Target и LeaveOneOut кодировщиков
    
    Returns:
        pd.DataFrame: DataFrame с закодированными признаками
    
    Поддерживаемые кодировщики:
        - 'OneHot': One-Hot Encoding (удаляет первую категорию для избежания мультиколлинеарности)
        - 'Label': Label Encoding (преобразует категории в числа 0, 1, 2, ...)
        - 'Ordinal': Ordinal Encoding (аналогично Label, но с явным указанием порядка)
        - 'Count': Count Encoding (заменяет категории на частоту их появления)
        - 'Binary': Binary Encoding (представляет категории в двоичном виде)
        - 'Frequency': Frequency Encoding (заменяет категории на их относительную частоту)
        - 'Hashing': Hash Encoding (использует хеширование для создания признаков)
        - 'BaseN': Base-N Encoding (представляет категории в системе счисления по основанию N)
        - 'Target': Target Encoding (заменяет категории на средние значения целевой переменной)
        - 'LeaveOneOut': Leave-One-Out Encoding (Target Encoding с защитой от переобучения)
    
    Example:
        >>> encoder_config = {
        ...     'OneHot': ['color', 'size'],
        ...     'Label': ['category'],
        ...     'Target': ['region'],
        ...     'Count': ['brand']
        ... }
        >>> encoded_df = encode_data(df, encoder_config, y=target_series)
    
    Raises:
        ValueError: Если для Target или LeaveOneOut кодировщиков не передан параметр y
        ValueError: Если указан неподдерживаемый тип кодировщика
    
    Note:
        - Неиспользованные колонки сохраняются в исходном виде
        - Порядок колонок: сначала оригинальные (неизмененные), затем новые закодированные
        - Функция создает новые колонки с суффиксами (_label, _ordinal, _count, _freq)
    """
    
    result = pd.DataFrame(index=df.index)
    used_columns = []
    
    for encoder_name, columns in encoder_columns_dict.items():
        if not columns:
            continue
            
        if encoder_name == 'OneHot':
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            
            for col in columns:
                transformed = ohe.fit_transform(df[[col]])
                ohe_cols = [f'{col}_{cat}' for cat in ohe.categories_[0][1:]]
                
                result = pd.concat([result, pd.DataFrame(transformed, columns=ohe_cols, index=df.index)], axis=1)
                used_columns.append(col)
                
        elif encoder_name == 'Label':
            le = LabelEncoder()
            for col in columns:
                result[f"{col}_label"] = le.fit_transform(df[col])
                used_columns.append(col)
        
        elif encoder_name == 'Ordinal':
            oe = OrdinalEncoder()
            for col in columns:
                result[f"{col}_ordinal"] = oe.fit_transform(df[[col]]).astype(int)
                used_columns.append(col)
                
        elif encoder_name == 'Count':
            for col in columns:
                cnt = df[col].value_counts()  # ИСПРАВЛЕНО: value_counts() вместо value_count()
                result[f"{col}_count"] = df[col].map(cnt)
                used_columns.append(col)
        
        elif encoder_name == 'Binary':
            be = ce.BinaryEncoder(cols=columns)  # ИСПРАВЛЕНО: = вместо -
            transformed = be.fit_transform(df[columns])
            result = pd.concat([result, transformed], axis=1)
            used_columns.extend(columns)
        
        elif encoder_name == 'Frequency':
            for col in columns:
                freq = df[col].value_counts(normalize=True)  # ИСПРАВЛЕНО: value_counts() вместо value_count()
                result[f'{col}_freq'] = df[col].map(freq)
                used_columns.append(col)
        
        elif encoder_name == 'Hashing':
            he = ce.HashingEncoder(cols=columns)  # ИСПРАВЛЕНО: HashingEncoder вместо BinaryEncoder
            transformed = he.fit_transform(df[columns])  # ИСПРАВЛЕНО: he вместо be
            result = pd.concat([result, transformed], axis=1)
            used_columns.extend(columns)
            
        elif encoder_name == 'BaseN':
            bne = ce.BaseNEncoder(cols=columns, base=3)
            transformed = bne.fit_transform(df[columns], y)  # ИСПРАВЛЕНО: transformed вместо tranformed
            result = pd.concat([result, transformed], axis=1)
            used_columns.extend(columns)
            
        elif encoder_name == 'Target':
            if y is None:
                raise ValueError('TargetEncoder требует параметр y')
                
            te = ce.TargetEncoder(cols=columns)
            transformed = te.fit_transform(df[columns], y)
            result = pd.concat([result, transformed], axis=1)
            used_columns.extend(columns)
        
        elif encoder_name == 'LeaveOneOut':
            if y is None:
                raise ValueError('LeaveOneOutEncoder требует параметр y')
            
            loo = ce.LeaveOneOutEncoder(cols=columns)
            transformed = loo.fit_transform(df[columns], y)
            result = pd.concat([result, transformed], axis=1)
            used_columns.extend(columns)  # ИСПРАВЛЕНО: used_columns вместо user_columns
            
        else:
            raise ValueError(f'Кодировщик {encoder_name} не поддерживается')  # ИСПРАВЛЕНО: опечатка
        
    # Добавляем неиспользованные колонки
    untouched_cols = [col for col in df.columns if col not in used_columns]
    if untouched_cols:
        result = pd.concat([result, df[untouched_cols]], axis=1)
        
    # Переупорядочиваем колонки: сначала оригинальные, затем новые
    result = result[[col for col in result.columns if col in df.columns] +
                   [col for col in result.columns if col not in df.columns]]
    
    return result


def get_encoding_info(df, categorical_columns):
    """
    Анализирует категориальные колонки и предлагает подходящие методы кодирования.
    
    Args:
        df (pd.DataFrame): DataFrame для анализа
        categorical_columns (list): Список категориальных колонок
    
    Returns:
        dict: Рекомендации по кодированию для каждой колонки
    """
    encoding_recommendations = {}
    
    for col in categorical_columns:
        unique_count = df[col].nunique()
        total_count = len(df)
        cardinality_ratio = unique_count / total_count
        
        recommendations = []
        
        if unique_count == 2:
            recommendations.append("Label (для бинарных признаков)")
        elif unique_count <= 10:
            recommendations.append("OneHot (малая кардинальность)")
        elif unique_count <= 50:
            recommendations.extend(["Label", "Count", "Binary"])
        else:
            recommendations.extend(["Hashing", "Target", "Binary"])
            
        if cardinality_ratio > 0.5:
            recommendations.append("Frequency (высокая кардинальность)")
            
        encoding_recommendations[col] = {
            'unique_values': unique_count,
            'cardinality_ratio': round(cardinality_ratio, 3),
            'recommended_encoders': recommendations
        }
    
    return encoding_recommendations
