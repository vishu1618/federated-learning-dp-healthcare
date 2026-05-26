import math


def approximate_epsilon(
    noise_multiplier,
    steps,
    sample_rate,
    delta=1e-5
):
    if noise_multiplier == 0:
        return float("inf")

    return (
        sample_rate
        * math.sqrt(2 * steps * math.log(1 / delta))
        / noise_multiplier
    )