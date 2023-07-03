from collections import Counter
import numpy as np
from scipy.stats import entropy as shannon
from scipy.stats import gaussian_kde


def shannon_entropy_kde(data: np.array):
    """
    Args:
        data (Numpy array): The dataset you want to calculate the shannon entropy for

    Returns:
        float: The calculated shannon entropy
    """
    # ajouter des exceptions sur shape
    try :
        num_columns = data.shape[1]
    except IndexError:
        num_columns = 1
    try :
        n_rows = data.shape[0]
    except IndexError:
        n_rows = 1

    entropies = []

    for i in range(num_columns):
        column_data = data[:, i]

        if np.issubdtype(column_data.dtype, np.number):
            kde = gaussian_kde(column_data)
            pk_distribution = kde(column_data)
            entropy_i = shannon(pk_distribution) / np.log2(n_rows)
        else:
            counter = Counter(column_data)
            total_count = len(column_data)
            pk_distribution = [count / total_count for count in counter.values()]
            entropy_i = shannon(pk_distribution) / np.log2(n_rows)

        entropies.append(entropy_i)
    entropy = np.array(entropies).mean()

    return entropy
