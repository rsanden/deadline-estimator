#!/usr/bin/env python3

import sys, os
import argparse
import re
import pickle
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import differential_evolution

# Provide time estimates at given confidence levels
# to calculate final time distribution.
#
# For example, given:
#
#   Task A:
#     68% chance task will be finished within the next 2 days
#     95% chance task will be finished within the next 5 days
#
#   Task B:
#     68% chance task will be finished within the next 3 days
#     95% chance task will be finished within the next 7 days
#
# We would have:
#
#   confidence_levels = [
#       0.68,
#       0.95,
#   ]
#   All tasks share these confidence_levels.
#
#   time_threshold_pairs = [
#       (2.0, 5.0),  # Task 1
#       (3.0, 7.0),  # Task 2
#   ]
#   Each task gets one pair in time_threshold_pairs.
#
# Regarding the deadline (days) CLI argument,
# this represents the amount of time left before a deadline,
# expressed in terms of effective man-days available to work.
# It may make sense to use a combined labor calculation here.

# Tasks are modeled as lognorm distributions,
# which are characterized by shape and scale parameters.
# parameter_bounds determines the acceptible range
# for these parameters

parameter_bounds = [
    (0.0001, 50.0),  # lognorm shape range
    (0.0001, 50.0),  # lognorm scale range
]

# Sample size for final result distribution
N = 2000000

# Threshold for confidence level comparison
SEARCH_THRESHOLD = 0.0001

# Histogram bin count
NUM_BINS = 100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", help="Don't plot anything (default: plot PDF)")
    parser.add_argument("--plot-cdf", action="store_true", help="plot result CDF instead of PDF")
    parser.add_argument(
        "--no-plot-deadline",
        action="store_true",
        help="Don't plot vertical deadline on PDF and CDF (default: Plot deadline)",
    )
    parser.add_argument(
        "--plot-term",
        type=str,
        default="wxt",
        help="Gnuplot terminal (ex: 'wxt', 'qt', 'x11', 'dumb size 128,48', 'png size 800,450', 'svg') (default: 'wxt')",
    )
    parser.add_argument("estimate_fpath", type=str, help="estimate filepath")
    parser.add_argument(
        "deadline_mandays",
        type=float,
        help="Deadline (effective-man-days from now) to calculate final on-time probability",
    )
    return parser.parse_args(sys.argv[1:])


def parse_estimate(fpath):
    """
    === Example estimate file ===

    Confidence A: 68.0%
    Confidence B: 95.0%
                               Estimate A    Estimate B
    ==================================================
     -- Task 1                    3.00          5.00
     -- Task 2                    1.00          3.00
         -- Subtask 2.1           0.25          0.50
    """
    task_descs = []
    confidence_levels = []
    time_threshold_pairs = []

    with open(fpath, "r") as fp:
        for line in fp:
            line = line.rstrip()
            if not line.strip() or line.strip().startswith("#"):
                continue

            # Headers
            m = re.match(r"^Estimate A\s+Estimate B$", line.strip())
            if m is not None:
                continue

            m = re.match(r"^=+$", line.strip())
            if m is not None:
                continue

            # Confidence levels
            m = re.match(r"^Confidence (\w+): ([\d.]+)%", line)
            if m is not None:
                if m.group(1) in ["A", "B"]:
                    confidence_level = float(m.group(2)) / 100.0
                    confidence_levels.append(confidence_level)
                    continue

            # Tasks
            if line.lstrip().startswith("-- "):
                m = re.match(r"^(.+)\s+([\d.]+)\s+([\d.]+)$", line)
                if m is not None:
                    task_desc = m.group(1).rstrip()
                    time_threshold_a = float(m.group(2))
                    time_threshold_b = float(m.group(3))
                    time_threshold_pair = (time_threshold_a, time_threshold_b)
                    task_descs.append(task_desc)
                    time_threshold_pairs.append(time_threshold_pair)
                    continue

            raise ValueError(f"Unrecognized line: {line}")

    if len(confidence_levels) != 2:
        raise ValueError(
            f"Invalid estimate file. Found {len(confidence_levels)} confidence levels"
        )

    if len(task_descs) != len(time_threshold_pairs):
        raise ValueError(
            f"Invalid estimate file. Found {len(task_descs)} tasks with {len(time_threshold_pairs)} threshold pairs"
        )

    return confidence_levels, task_descs, time_threshold_pairs


def get_cost_fn(time_threshold_pairs, confidence_levels):
    def cost_fn(params):
        eq1 = (
            lognorm.cdf(time_threshold_pairs[0], params[0], scale=params[1])
            - confidence_levels[0]
        )
        eq2 = (
            lognorm.cdf(time_threshold_pairs[1], params[0], scale=params[1])
            - confidence_levels[1]
        )
        return eq1 ** 2 + eq2 ** 2

    return cost_fn


def find_distribution(time_threshold_pair):
    while True:
        cost_fn = get_cost_fn(time_threshold_pair, confidence_levels)
        solution = differential_evolution(cost_fn, parameter_bounds)
        if not solution.success:
            raise RuntimeError(f"time_threshold_pairs failed: {time_threshold_pairs}")

        shape = solution.x[0]
        scale = solution.x[1]

        time_a = time_threshold_pair[0]
        prob_a = lognorm.cdf(time_a, shape, scale=scale)
        confidence_level_a = confidence_levels[0]

        time_b = time_threshold_pair[1]
        prob_b = lognorm.cdf(time_b, shape, scale=scale)
        confidence_level_b = confidence_levels[1]

        if (
            abs(prob_a - confidence_level_a) < SEARCH_THRESHOLD
            and abs(prob_b - confidence_level_b) < SEARCH_THRESHOLD
        ):
            return (shape, scale, prob_a, prob_b)

        print(
            f"Search failed: shape={shape:0.4f}, scale={scale:0.4f}, p({time_a:05.2f})={prob_a*100:0.2f}%, p({time_b:05.2f})={prob_b*100:0.2f}%"
        )


def plot_pdf_histogram(totals, num_bins=NUM_BINS, deadline=None, gnuplot_term="wxt"):
    import gnuplotlib as gp

    start = 0.0
    end = 2.0 * max(np.median(totals), np.mean(totals))
    bin_width = (end - start) / num_bins
    n, pdf_x = np.histogram(totals, bins=np.linspace(start, end, num=num_bins + 1))
    pdf_y = n / (len(totals) * bin_width)

    plots = []
    plot_pdf_outline = (
        pdf_x[:-1],
        pdf_y,
        {"with": 'boxes lc rgb "light-blue" fill solid'},
    )
    plot_pdf_fill = (
        pdf_x[:-1],
        pdf_y,
        {"with": 'lines lc rgb "blue" lw 2'},
    )
    plots.append(plot_pdf_outline)
    plots.append(plot_pdf_fill)

    if deadline is not None:
        plot_deadline = (
            np.array([deadline, deadline]),
            np.array([0, max(pdf_y)]),
            {"with": 'lines lc rgb "red" lw 1'},
        )
        plots.append(plot_deadline)

    kwargs = {}
    if gnuplot_term.startswith("png"):
        kwargs["output"] = "deadline_estimate_pdf.png"
    if gnuplot_term.startswith("gif"):
        kwargs["output"] = "deadline_estimate_pdf.gif"
    if gnuplot_term.startswith("jpeg"):
        kwargs["output"] = "deadline_estimate_pdf.jpg"
    if gnuplot_term.startswith("svg"):
        kwargs["output"] = "deadline_estimate_pdf.svg"
    gp.plot(
        *plots,
        title="PDF of Project Completion Date",
        xlabel="Man-Days",
        ylabel="Probability Density",
        xrange=(start, end),
        terminal=gnuplot_term,
        **kwargs,
    )
    if not any(
        gnuplot_term.startswith(term) for term in ["dumb", "png", "gif", "jpeg", "svg"]
    ):
        gp.wait()


def plot_cdf_histogram(totals, num_bins=NUM_BINS, deadline=None, gnuplot_term="wxt"):
    import gnuplotlib as gp

    start = 0.0
    end = 2.0 * max(np.median(totals), np.mean(totals))
    cdf_x = np.linspace(start, end, num=num_bins)
    cdf_y = np.searchsorted(totals, cdf_x, side="right") / len(totals)

    plots = []
    plot_cdf_outline = (
        cdf_x,
        cdf_y,
        {"with": 'lines lc rgb "blue" lw 2'},
    )
    plots.append(plot_cdf_outline)

    if deadline is not None:
        plot_deadline = (
            np.array([deadline, deadline]),
            np.array([0, max(cdf_y)]),
            {"with": 'lines lc rgb "red" lw 1'},
        )
        plots.append(plot_deadline)

    kwargs = {}
    if gnuplot_term.startswith("png"):
        kwargs["output"] = "deadline_estimate_cdf.png"
    if gnuplot_term.startswith("gif"):
        kwargs["output"] = "deadline_estimate_cdf.gif"
    if gnuplot_term.startswith("jpeg"):
        kwargs["output"] = "deadline_estimate_cdf.jpg"
    if gnuplot_term.startswith("svg"):
        kwargs["output"] = "deadline_estimate_cdf.svg"
    gp.plot(
        *plots,
        title="CDF of Project Completion Date",
        xlabel="Man-Days",
        ylabel="Cumulative Probability",
        xrange=(start, end),
        terminal=gnuplot_term,
        **kwargs,
    )
    if not any(
        gnuplot_term.startswith(term) for term in ["dumb", "png", "gif", "jpeg", "svg"]
    ):
        gp.wait()


if __name__ == "__main__":
    args = get_args()
    confidence_levels, task_descs, time_threshold_pairs = parse_estimate(
        args.estimate_fpath
    )

    found_distributions = {}
    if os.path.isfile("distributions_cache.pickle"):
        with open("distributions_cache.pickle", "rb") as fp:
            found_distributions = pickle.load(fp)

    lognorm_vars = []
    for i, time_threshold_pair in enumerate(time_threshold_pairs):
        distribution_key = (tuple(confidence_levels), time_threshold_pair)
        if distribution_key not in found_distributions:
            found_distributions[distribution_key] = find_distribution(
                time_threshold_pair
            )
        time_a, time_b = time_threshold_pair
        shape, scale, prob_a, prob_b = found_distributions[distribution_key]
        lognorm_vars.append((shape, scale))
        print(
            f"Task {i:03d}: shape={shape:0.4f}, scale={scale:0.4f}, p({time_a:05.2f})={prob_a*100:0.2f}%, p({time_b:05.2f})={prob_b*100:0.2f}%"
        )

    with open("distributions_cache.pickle", "wb") as fp:
        pickle.dump(found_distributions, fp)

    totals = np.zeros(N)
    for lognorm_var in lognorm_vars:
        shape, scale = lognorm_var
        totals += np.random.lognormal(mean=np.log(scale), sigma=shape, size=N)
    totals = np.sort(totals)

    agg_median = np.median(totals)
    agg_mean = np.mean(totals)
    agg_stdev = np.var(totals) ** 0.5
    print(f"Aggregate median: {agg_median:0.4f}")
    print(f"Aggregate mean:   {agg_mean:0.4f}")
    print(f"Aggregate stdev:  {agg_stdev:0.4f}")

    agg_prob = np.count_nonzero(totals < args.deadline_mandays) / float(N)
    print(
        f"Probability to complete within {args.deadline_mandays:0.2f} effective man-days: {agg_prob*100:0.2f}%"
    )

    if not args.no_plot:
        if args.plot_cdf:
            plot_cdf_histogram(
                totals,
                num_bins=NUM_BINS,
                deadline=args.deadline_mandays if not args.no_plot_deadline else None,
                gnuplot_term=args.plot_term,
            )
        else:
            plot_pdf_histogram(
                totals,
                num_bins=NUM_BINS,
                deadline=args.deadline_mandays if not args.no_plot_deadline else None,
                gnuplot_term=args.plot_term,
            )
