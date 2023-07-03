import numpy as np
import pandas as pd
from entropies import shannon_entropy_kde


def initialiser_dictionnaire(liste_cles: list):
    """_summary_
    This function takes in argument a list of keys
    and initialize a python dictionnary with 0 for each key.
    ==============================================================================================
    Args:
        liste_cles (list): The list of keys you want to initialize your python dictionnary with.

    Returns:
        dict: The initialized dictionnary
    """
    dic = {key: 0 for key in liste_cles}
    return dic


def sampling_kg(df: pd.DataFrame, players: dict, k: int, id_columns: list = None) :
    """_summary_
    This function perform the contribution of the participants of a knowle
    ============================================================================================
    Args:
        df (pd.DataFrame): The coalition dataset
        players (dict): The players of the coalition dict in the form {player_name: [features]}.
        k (int): Number of iteration to perform Shapley value approximation
        id_columns: Idendification column

    Returns:
        dict: _description_
    """
    date_columns = df.select_dtypes(include='datetime64').columns.tolist()
    text_columns = df.select_dtypes(include='object').columns.tolist()
    num_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    try:
        dataset = df.copy().drop(date_columns + id_columns, axis=1)
    except TypeError:
        dataset = df.copy().drop(date_columns, axis=1)
    dataset[text_columns] = dataset[text_columns].fillna(method='backfill', axis=1)
    dataset[num_columns] = dataset[num_columns].fillna(value=9999999999999, axis=1)
    ply = list(players.keys())
    m = len(ply)
    shapley_value = initialiser_dictionnaire(ply)

    # for each sample
    for _ in range(k):
        coalition = np.random.choice(ply, size=np.random.randint(2, m + 1), replace=False)
        features = []
        for player in coalition :
            features += players[player]
        coalition_av = shannon_entropy_kde(dataset[features].values)  # av=accuracy value

        for player in coalition :
            coalition_without_player = [p for p in coalition if p != player]
            features_wh_player = []
            for player in coalition_without_player :
                features_wh_player += players[player]
            dataset_ft_wp_val = dataset[features_wh_player].values
            coalition_wp_av = shannon_entropy_kde(dataset_ft_wp_val)  # wp=without player
            contribution = coalition_av - coalition_wp_av

            shapley_value[player] += contribution / k
    return shapley_value
