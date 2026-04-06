import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import numpy as np

def plot_hist_boxplot(data, columns, bins=17, path=None):
    """
    Визуализирует распределение одной или нескольких числовых колонок.
    
    Параметры:
    -----------
    data : pandas.DataFrame
        DataFrame с данными
    columns : str или list
        Название колонки или список названий колонок для анализа
    bins : int, optional
        Количество бинов для гистограммы (по умолчанию 17)
    path : str, optional
        Путь для сохранения графика (если None, график только отображается)
    """
    
    # Нормализуем входные данные: преобразуем строку в список
    if isinstance(columns, str):
        columns = [columns]
    
    n = len(columns)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2*n))
    
    # Если только одна колонка, axes становится 1D массивом - преобразуем в 2D
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        # Получаем данные колонки
        col_data = data[col].copy()
        
        # Обрабатываем бесконечности
        col_data = col_data.replace([np.inf, -np.inf], np.nan)
        
        # Удаляем NaN
        col_data = col_data.dropna()
                
        # Столбец 0: Гистограмма
        sns.histplot(col_data, ax=axes[i, 0], bins=bins, kde=False, color='salmon', edgecolor='white')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlabel(col)
        
        # Столбец 1: Boxplot
        boxplot = axes[i, 1].boxplot(col_data, vert=False, patch_artist=True, widths=0.4)
        # Раскрашиваем boxplot
        boxplot['boxes'][0].set_facecolor('skyblue')
        boxplot['boxes'][0].set_alpha(0.7)
        
        axes[i, 1].grid(True, alpha=0.3, axis='x')
        axes[i, 1].set_xlabel(col)
        axes[i, 1].set_yticks([])  # Убираем метки на оси Y для boxplot
        
        # Столбец 2: Q-Q plot
        try:
            (osm, osr), (slope, intercept, r) = stats.probplot(col_data, dist="norm", fit=True)
            axes[i, 2].plot(osm, osr, 'o', color='skyblue', alpha=0.7, markersize=4)
            axes[i, 2].plot(osm, slope * osm + intercept, color='salmon', linewidth=2)
            axes[i, 2].set_xlabel(col)
            
        except Exception as e:
            axes[i, 2].text(0.5, 0.5, f'Ошибка Q-Q plot:\n{str(e)}', 
                           ha='center', va='center', transform=axes[i, 2].transAxes)
            print(f"Ошибка при построении Q-Q plot для '{col}': {e}")
        
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)  # Увеличиваем отступы для лучшего отображения
    
    if path:
        plt.savefig(path, bbox_inches='tight', dpi=300)
        print(f"График сохранен: {path}")
    
    plt.show()