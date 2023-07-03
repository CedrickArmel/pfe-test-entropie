from scipy.stats import gaussian_kde
from scipy.stats import entropy as shannon
import numpy as np
from collections import Counter


def shannon_entropy_kde(data):
    ##ajouter des exceptions sur shape
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
            pk = kde(column_data)
            ei = shannon(pk)/np.log2(n_rows)
        else:
            counter = Counter(column_data)
            total_count = len(column_data)
            pk = [count / total_count for count in counter.values()]
            ei = shannon(pk)/np.log2(n_rows)

        entropies.append(ei)
    
    entropy = np.array(entropies).mean()

    return entropy
