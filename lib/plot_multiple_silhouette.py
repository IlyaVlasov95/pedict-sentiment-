from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time

def plot_silhouette(data, n_range, n_cols,  path=None, n_jobs = -1):

    start_time = time.time()
    n_plots = len(n_range)
    n_rows = math.ceil(n_plots / n_cols)
    
    def calculate_silhouette(n_clusters):
        model = KMeans(
            n_clusters = n_clusters,
            random_state = 42,
            init = 'k-means++'
        )
        
        cluster_labels = model.fit_predict(data)       
        silhouette_avg = silhouette_score(data, cluster_labels)
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        
        return {
            'n_clusters': n_clusters,
            'silhouette_avg': silhouette_avg,
            'sample_silhouette_values': sample_silhouette_values,
            'cluster_labels': cluster_labels
        }
    
    results = Parallel(n_jobs = n_jobs)(
        delayed(calculate_silhouette)(n_clusters)
        for n_clusters in n_range
    )
        
        
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, result in enumerate(results):
        ax = axes[i]
        n_clusters = result['n_clusters']
        silhouette_avg = result['silhouette_avg']
        sample_silhouette_values = result['sample_silhouette_values']
        cluster_labels = result['cluster_labels']
    
        y_lower = 10
        ax.set_title(f'Silhouette plot for k={n_clusters} (avg={silhouette_avg:.2f})', fontsize=12)
        ax.set_xlabel('Silhouette coefficient values')
        ax.set_ylabel('Cluster label')

        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        #print(n_clusters)
        for j in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
            ith_cluster_silhouette_values.sort()
            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = cm.nipy_spectral(float(j) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color='red', linestyle='--')
        ax.grid(True, alpha=0.3)
        


    # Удаляем пустые графики
    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if path:
        save_path = f'{path}/silhouette_plots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')    
    
    #print(f'Графики сохранены в файл: {save_path}')
    plt.show()
    
    end_time = time.time()
    print(f"⚡ Время выполнения: {(end_time - start_time)/60:.0f} минут")
