#!/usr/bin/env python3
import os
import numpy as np
import pickle

KIND_GENERAL = 0
KIND_HSDIR = 1
KIND_INTRO = 2
KIND_REND = 3

root = "results"
kind_counter = {}
kind_counter[KIND_GENERAL] = 0
kind_counter[KIND_HSDIR] = 0
kind_counter[KIND_INTRO] = 0
kind_counter[KIND_REND] = 0

def save_trace(cells, kind):
    times = cells["time"]
    directions = cells["direction"]
    assert len(times) == len(directions)
    assert kind in [KIND_GENERAL, KIND_HSDIR, KIND_INTRO, KIND_REND]
    if len(times) == 0:
        return
    start_time = times[0]

    i = 0
    for d in directions:
        if d != 0:
            i += 1
        else:
            break
    if i < 30:
        return

    # only save 10000 traces per kind
    i = kind_counter[kind]
    if i >= 10000:
        return
    kind_counter[kind] += 1

    filename = f"{root}/{kind}/{i}.log"
    lines = 0
    with open(filename, "w") as f:
        for time, direction in zip(times, directions):
            if direction == 0:
                # done with this circuit
                return
            d = "s" if direction == 1 else "r"
            # convert from float s to nanoseconds
            t = int((time - start_time) * 1e9)
            f.write(f"{t},{d},514\n")
            # only save the first 30 lines
            lines += 1
            if lines >= 30:
                break


def extract_circuits_to_kind_traces(files, name):
    for i, f in enumerate(files):
        with open(os.path.join(name, f), "rb") as file:
            (tag, selected_circuits) = pickle.load(file)
            for circuit in selected_circuits:
                kind = circuit["kind"]
                save_trace(
                    circuit["cells"][np.vectorize(lambda cell: cell[3] != 16)(circuit["cells"])],
                    kind
                )


def get_pickle_list(dir):
    return sorted([f for f in os.listdir(dir) if f.endswith(".pickle")])

# check if the results folder already exists
if os.path.exists(root):
    print("results folder already exists")
    exit(1)
os.mkdir(root)

# subfolder for each kind, 0=general, 1=hsdir, 2=intro, 3=rend
print(f"created {root} folder")
for i in range(4):
    os.mkdir(os.path.join(root, str(i)))
    print(f"created {root}/{i} folder")

# we use the autoloc dataset, since it has plenty of general and rend circuits
print("listing files...")
autoloc_files = get_pickle_list("autoloc")

print("extracting circuits to kind traces...")
extract_circuits_to_kind_traces(autoloc_files, "autoloc")

print("done")
