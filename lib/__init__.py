# lib/__init__.py

# Импортируем функции, которые хотим сделать доступными
from lib.visualization import plot_hist_boxplot
from lib.sample_data import sample_data_agg
from lib.data_scaler import normalize_data_with_scalers
from lib.cluster_analys import parallel_kmeans_cluster_analysis
from lib.plot_cluster import plot_cluster_cat
from lib.plot_multiple_silhouette import plot_silhouette

# Определяем, что будет импортироваться при from lib import *
__all__ = [
    'plot_hist_boxplot',
    'sample_data_agg',
    'normalize_data_with_scaler',
    'parallel_kmeans_cluster_analysis',
    'plot_cluster_cat',
    'plot_silhouette'
]
