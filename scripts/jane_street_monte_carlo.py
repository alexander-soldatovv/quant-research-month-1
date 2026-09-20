import math
from fractions import Fraction

import numpy as np


SEED = 20260920
N = 2_000_000


def ci95_for_mean(samples):
    mean = samples.mean()
    se = samples.std(ddof=1) / math.sqrt(len(samples))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def simulate_max_of_three_d6(rng):
    rolls = rng.integers(1, 7, size=(N, 3))
    return rolls.max(axis=1)


def simulate_optional_reroll(rng):
    first = rng.integers(1, 7, size=N)
    second = rng.integers(1, 7, size=N)
    return np.where(first <= 3, second, first)


def simulate_flips_to_three_heads(rng):
    results = np.empty(N, dtype=np.int32)
    for trial in range(N):
        streak = 0
        flips = 0
        while streak < 3:
            flips += 1
            if rng.random() < 0.5:
                streak += 1
            else:
                streak = 0
        results[trial] = flips
    return results


def simulate_sum_product_dependence(rng):
    rolls = rng.integers(1, 7, size=(N, 2))
    sums = rolls[:, 0] + rolls[:, 1]
    products = rolls[:, 0] * rolls[:, 1]

    event_s2 = sums == 2
    event_p1 = products == 1
    joint = event_s2 & event_p1

    return {
        "p_sum_2": event_s2.mean(),
        "p_product_1": event_p1.mean(),
        "p_joint": joint.mean(),
        "p_product_1_given_sum_2": event_p1[event_s2].mean(),
        "independent_joint_estimate": event_s2.mean() * event_p1.mean(),
    }


def print_mean_result(name, samples, exact):
    mean, lo, hi = ci95_for_mean(samples)
    print(name)
    print(f"  Monte Carlo mean: {mean:.6f}")
    print(f"  95% CI:           [{lo:.6f}, {hi:.6f}]")
    print(f"  Analytic value:   {float(exact):.6f} ({exact})")
    print(f"  Error:            {mean - float(exact):+.6f}")
    print()


def main():
    rng = np.random.default_rng(SEED)

    max_samples = simulate_max_of_three_d6(rng)
    print_mean_result("1. Maximum of 3 d6 rolls", max_samples, Fraction(119, 24))

    reroll_samples = simulate_optional_reroll(rng)
    print_mean_result("2. Optional one-time d6 reroll", reroll_samples, Fraction(17, 4))

    three_heads_samples = simulate_flips_to_three_heads(rng)
    print_mean_result("3. Flips until HHH", three_heads_samples, Fraction(14, 1))

    dependence = simulate_sum_product_dependence(rng)
    print("4. Dependence of B+R and B*R")
    print(f"  P(S = 2):                 {dependence['p_sum_2']:.6f} analytic {1/36:.6f}")
    print(f"  P(P = 1):                 {dependence['p_product_1']:.6f} analytic {1/36:.6f}")
    print(f"  P(S = 2 and P = 1):       {dependence['p_joint']:.6f} analytic {1/36:.6f}")
    print(f"  P(P = 1 | S = 2):         {dependence['p_product_1_given_sum_2']:.6f} analytic 1.000000")
    print(f"  If independent, joint ~=  {dependence['independent_joint_estimate']:.6f} analytic {1/1296:.6f}")
    print()
    print("Since the joint probability is close to 1/36, not 1/1296, S and P are not independent.")


if __name__ == "__main__":
    main()
