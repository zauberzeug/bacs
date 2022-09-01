#!/usr/bin/env python3
from bacs import bacs
from input_output import read_input, write_output

# %% usage example
dataset = 'parkinglot_near'
l, Sll, ijt, Xa, Ma, P = read_input(dataset)
ld, Xd, Md, Sdd, s0dsq, vr, w, iteration = bacs(l, ijt, Sll, Xa, Ma, P)
write_output(dataset, ld, Xd, Md, Sdd, vr, w)

# %% usage example with points at infinity
dataset = 'parkinglot_far'
l, Sll, ijt, Xa, Ma, P = read_input(dataset)
ld, Xd, Md, Sdd, s0dsq, vr, w, iteration = bacs(l, ijt, Sll, Xa, Ma, P, near_ratio=0.5)
write_output(dataset, ld, Xd, Md, Sdd, vr, w)

# %% usage expample with points at infinity and outliers
dataset = 'parkinglot_outlier'
l, Sll, ijt, Xa, Ma, P = read_input(dataset)
ld, Xd, Md, Sdd, s0dsq, vr, w, iteration = bacs(l, ijt, Sll, Xa, Ma, P, near_ratio=0.5, k=3)
write_output(dataset, ld, Xd, Md, Sdd, vr, w)
