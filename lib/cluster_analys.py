from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

def parallel_kmeans_cluster_analysis(data, k_range, path=None, n_jobs=-1):
    """
    Комбинированный подход: KMeans + параллельная обработка + прогресс-бар + точное время
    
    Args:
        data: данные для кластеризации (numpy array или pandas DataFrame)
        k_range: диапазон значений K (tuple: (min_k, max_k))
        path: путь для сохранения графиков (опционально)
        n_jobs: количество ядер для параллельных вычислений (-1 = все ядра)
        
    Returns:
        tuple: (optimal_k_elbow, optimal_k_silhouette)
    """
    start_time = time.perf_counter()  # Точный таймер
    
    # Создаем фигуру для двух графиков
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    
    def calculate_metrics(k):
        """Вычисляет метрики для одного значения K"""
        model = KMeans(
            n_clusters=k,
            random_state=42,
            init='k-means++',
            n_init='auto'  # Автоматический выбор количества инициализаций
        )
        
        model.fit(data)
        distortion = -model.score(data)  # Сумма квадратов расстояний
        sil_score = silhouette_score(data, model.labels_)
        
        return k, distortion, sil_score
    
    # Параллельные вычисления с прогресс-баром
    print(f"Вычисление метрик для K от {k_range[0]} до {k_range[1]}...")
    metrics = Parallel(n_jobs=n_jobs)(
        delayed(calculate_metrics)(k)
        for k in tqdm(range(k_range[0], k_range[1] + 1), 
                      desc="Анализ кластеров")
    )
    
    # Разделяем результаты
    k_values = [m[0] for m in metrics]
    distortions = [m[1] for m in metrics]
    sil_scores = [m[2] for m in metrics]
    
    def find_elbow(values, k_values):
        """Находит точку 'локтя' на кривой"""
        diffs = np.diff(values)
        return k_values[np.argmax(diffs) + 1]
    
    def plot_metric(ax, values, k_values, metric_name, optimal_value=None):
        """Визуализирует одну метрику"""
        ax.plot(k_values, values, 'bo-',color='black')
        ax.set_xlabel('Количество кластеров (K)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} метод')
        ax.grid(True, alpha=0.3)
        
        if optimal_value:
            ax.axvline(x=optimal_value, linestyle='--', color='red', 
                       label=f'Оптимальное K = {optimal_value}')
            ax.legend()
    
    # Находим оптимальные K
    elbow_k = find_elbow(distortions, k_values)
    best_sil_k = k_values[np.argmax(sil_scores)]
    
    # Строим графики
    plot_metric(axes[0], distortions, k_values, 'Distortion (локоть)', elbow_k)
    plot_metric(axes[1], sil_scores, k_values, 'Silhouette Score', best_sil_k)
    
    # Сохранение и отображение
    plt.tight_layout()
    if path:
        save_path = f'{path}/cluster_analysis.png' if not path.endswith('.png') else path
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Точное время выполнения
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    
    # Вывод результатов
    print("\nРезультаты кластерного анализа:")
    print(f"Общее время выполнения: {int(mins)} мин {secs:.2f} сек")
    print(f"Рекомендуемое K (метод локтя): {elbow_k}")
    print(f"Рекомендуемое K (silhouette score): {best_sil_k}")
    print(f"Использованная модель: KMeans(init='k-means++', n_init='auto')")
    
    return elbow_k, best_sil_k