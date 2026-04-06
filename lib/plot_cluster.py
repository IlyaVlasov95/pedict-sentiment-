import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_cluster_cat(data, labels, colors, columns_to_plot, N, 
                     path = None):
    """
    Улучшенная версия функции визуализации кластеров с отдельным сохранением графиков
    и общим subplot с сеткой 3 колонки
    """
    
    # Проверка входных данных
    if N >= len(columns_to_plot):
        raise ValueError(f"Индекс N={N} превышает количество колонок")
    
    n_clusters = len(set(labels))
    plot_columns = np.delete(columns_to_plot, N)
    n_plots = len(plot_columns)
    
    # 1. СОХРАНЯЕМ КАЖДЫЙ ГРАФИК ОТДЕЛЬНО
    print(f"Сохраняем {n_plots} отдельных графиков...")
    
    for i, col in enumerate(plot_columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Построение точек для каждого кластера
        for cluster in range(n_clusters):
            cluster_mask = labels == cluster
            cluster_points = data[cluster_mask]
            
            ax.scatter(
                cluster_points[columns_to_plot[N]],
                cluster_points[col],
                c=colors[cluster],
                label=f'Cluster {cluster}',
                s=20, alpha=0.7
            )
        
        # Оформление отдельного графика
        ax.set_xlabel(columns_to_plot[N], fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{col} vs {columns_to_plot[N]}', fontsize=14)
        
        # Легенда для отдельного графика
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colors[cluster], markersize=10)
            for cluster in range(n_clusters)
        ]
        labels_legend = [f"Cluster {cluster}" for cluster in range(n_clusters)]
        ax.legend(handles, labels_legend, fontsize=10)
        
        # Сохраняем отдельный график
        individual_save_path = f'{path}_{col}.png'
        plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Сохранен: {individual_save_path}")

    # 2. СОЗДАЕМ ОБЩИЙ SUBPLOT С СЕТКОЙ 3 КОЛОНКИ
    #print(f"\nСоздаем общий subplot с сеткой 3 колонки...")
    
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Правильная обработка axes для разных случаев
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    
    # Делаем axes плоским для удобства
    axes_flat = axes.flatten()

    # Строим графики в общем subplot
    for i, col in enumerate(plot_columns):
        ax = axes_flat[i]
        
        # Построение точек для каждого кластера
        for cluster in range(n_clusters):
            cluster_mask = labels == cluster
            cluster_points = data[cluster_mask]
            
            ax.scatter(
                cluster_points[columns_to_plot[N]],
                cluster_points[col],
                c=colors[cluster],
                label=f'Cluster {cluster}',
                s=15, alpha=0.7
            )
        
        # Оформление графика в subplot
        ax.set_xlabel(columns_to_plot[N], fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{col} vs {columns_to_plot[N]}', fontsize=11)

    # Удаляем пустые графики, если есть
    for j in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Общая легенда для subplot
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=colors[cluster], markersize=10)
        for cluster in range(n_clusters)
    ]
    labels_legend = [f"Cluster {cluster}" for cluster in range(n_clusters)]
    
    fig.legend(handles, labels_legend, loc='upper center', 
               ncol=n_clusters, fontsize=12, 
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    plt.suptitle('Cluster Analysis Results', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Сохраняем общий график
    if path:
        combined_save_path = f'{path}_combined.png'
        plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Общий график сохранен: {combined_save_path}")
    print(f"\n📊 Итого создано графиков: {n_plots} отдельных + 1 общий")

    return n_plots
