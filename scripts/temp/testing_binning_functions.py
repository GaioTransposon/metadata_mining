#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:03:22 2023

@author: dgaio
"""


import time
import random



def bin_numbers(numbers, max_sum):
    groups = []
    current_group = []
    current_sum = 0

    for number in numbers:
        if current_sum + number <= max_sum:
            current_group.append(number)
            current_sum += number
        else:
            groups.append(current_group)
            current_group = [number]
            current_sum = number

    # Add the last group if it's not empty
    if current_group:
        groups.append(current_group)

    return groups


def alternative_bin_numbers(numbers, max_sum):
    numbers.sort(reverse=True)  # Sort numbers in descending order
    bins = []

    for number in numbers:
        placed = False
        for bin in bins:
            if sum(bin) + number <= max_sum:
                bin.append(number)
                placed = True
                break
        if not placed:
            bins.append([number])

    return bins


def best_fit_bin_numbers(numbers, max_sum):
    bins = []

    for number in numbers:
        # Find the bin that will be closest to max_sum after adding this number
        best_fit = None
        min_space_left = max_sum
        for i, bin in enumerate(bins):
            space_left = max_sum - sum(bin)
            if number <= space_left < min_space_left:
                best_fit = i
                min_space_left = space_left

        if best_fit is None:
            # Create new bin if no suitable bin is found
            bins.append([number])
        else:
            # Add number to the best-fit bin
            bins[best_fit].append(number)

    return bins


def first_fit_decreasing_bin_numbers(numbers, max_sum):
    # Sort numbers in descending order
    sorted_numbers = sorted(numbers, reverse=True)
    bins = []

    for number in sorted_numbers:
        placed = False
        for bin in bins:
            if sum(bin) + number <= max_sum:
                bin.append(number)
                placed = True
                break
        if not placed:
            bins.append([number])

    return bins




# Generate a list of 100 random numbers between 1 and 5
numbers = [random.randint(1, 5) for _ in range(200)]

print(numbers)
max_sum = 5

# Timing the original function
start_time = time.time()
binned_groups = bin_numbers(numbers, max_sum)
end_time = time.time()
print("Original Binning Function:")
print("Binned Groups:", binned_groups)
print("bins:", len(binned_groups))
print("Execution Time:", end_time - start_time, "seconds")

# Timing the alternative function
start_time_alt = time.time()
binned_groups_alt = alternative_bin_numbers(numbers, max_sum)
end_time_alt = time.time()
print("\nAlternative Binning Function:")
print("Binned Groups:", binned_groups_alt)
print("bins:", len(binned_groups_alt))
print("Execution Time:", end_time_alt - start_time_alt, "seconds")




# Timing the Best Fit function
start_time_best = time.time()
binned_groups_best = best_fit_bin_numbers(numbers, max_sum)
end_time_best = time.time()
print("\nBest Fit Binning Function:")
print("Binned Groups:", binned_groups_best)
print("bins:", len(binned_groups_best))
print("Execution Time:", end_time_best - start_time_best, "seconds")




# Timing the First Fit Decreasing function
start_time_ffd = time.time()
binned_groups_ffd = first_fit_decreasing_bin_numbers(numbers, max_sum)
end_time_ffd = time.time()
print("\nFirst Fit Decreasing Binning Function:")
print("Binned Groups:", binned_groups_ffd)
print("bins:", len(binned_groups_ffd))
print("Execution Time:", end_time_ffd - start_time_ffd, "seconds")


