import random


def generate_white_noise(num_samples: int = 100, amplitude: float = 10.0):
    """_summary_

    Args:
        num_samples (int, optional): Defaults to 100.
        amplitude (float, optional): Defaults to 10.0.

    Returns:
        list
    """

    dataset = []
    for _ in range(num_samples):
        sample = random.uniform(-amplitude, amplitude)
        dataset.append(sample)
    return dataset
