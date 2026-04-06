import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def sample_data_agg(data, frac, agg_col, columns):
    # Создание профилей клиентов
    client_profiles = data.groupby(agg_col)[columns].mean()

    # Создание страт
    client_profiles['strata'] = client_profiles.apply(
        lambda row: "_".join([
            str(pd.qcut([row[col]], q=5, labels=False, duplicates='drop')[0])
            for col in columns]), axis=1)

    # Стратифицированная выборка
    sample_clients = (
        client_profiles
        .groupby('strata', group_keys=False)
        .apply(lambda x: x.sample(frac=frac, random_state=42))
    )

    sample_data = data[data[agg_col].isin(sample_clients.index)]

    # Определяем количество графиков и сетку 3 колонки
    n_cols = 3
    n_plots = len(columns)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    # Если всего один график, axes не массив, делаем его массивом
    if n_plots == 1:
        axes = np.array([[axes]])
    # Если одна строка, делаем axes двумерным для удобства
    elif n_rows == 1:
        axes = np.array([axes])

    # Проходим по колонкам и рисуем KDE на одном графике для полной и стратифицированной выборок
    for i, col in enumerate(columns):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx]

        # Рисуем обе выборки на одном графике
        sns.kdeplot(data[col], ax=ax, fill=True, alpha=0.5, color='salmon', label='Полная выборка')
        sns.kdeplot(sample_data[col], ax=ax, fill=True, alpha=0.5, color='skyblue', label='Стратифицированная выборка')

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel('Плотность')
        ax.legend()

    # Удаляем пустые графики, если есть
    total_subplots = n_rows * n_cols
    if total_subplots > n_plots:
        for j in range(n_plots, total_subplots):
            fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("Сравнение размеров выборок:")
    print(f"Полная выборка: {len(data)} строк, {data[agg_col].nunique()} уникальных клиентов")
    print(f"Стратифицированная выборка: {len(sample_data)} строк, {sample_data[agg_col].nunique()} уникальных клиентов")
    print(f"Доля выборки: {len(sample_data)/len(data)*100:.1f}%")

    return sample_data
