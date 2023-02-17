#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
import pandas as pd
import numpy.random as rd
import arbor as A
from time import perf_counter as pc
from logging import warning
import seaborn as sns
import matplotlib.pyplot as plt

def make_hh():
    # TODO figure out HH parameters
    # TODO figure out cell geometry
    tree = A.segment_tree()
    tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)
    center = "(location 0 0.5)"
    soma = "(tag 1)"
    decor = (
        A.decor()
        .set_property(Vm=-65)
        .paint(soma, A.density("hh", {'gkbar':3.6, 'gl': 0.03, 'gnabar': 12.0, 'el':-65}))
        .place(center, A.threshold_detector(-50), "source")
        .place(center, A.synapse("expsyn", {"tau": 0.5, "e": 0}), "synapse")
    )
    return A.cable_cell(tree, decor)

def make_iaf():
    res = A.lif_cell("source", "synapse")
    res.tau_m=10
    res.t_ref=2
    res.C_m=250
    res.E_L=-65
    res.V_m=-65
    res.E_R=-65
    return res

class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.f_background = 8e-3 # kHz
        # Indegree of background connection, used as a scale for the frequency here
        # TODO test/check if this holds water
        self.k_background = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100, 0])
        self.weight_background = 5
        # Thalamic inputs
        self.f_thalamic = 15e-3
        self.weight_thalamic = 5
        self.delay_thalamic = 1.5

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return make_hh()

    def global_properties(self, kind):
        return A.neuron_cable_properties()

    def event_generators(self, gid):
        pop = 0
        f = self.f_background*self.k_background[pop]
        return [
            A.event_generator(
                "synapse",
                self.weight_background,
                A.poisson_schedule(tstart=0.0, freq=f),
            )
        ]

    def probes(self, gid):
        return [A.cable_probe_membrane_voltage('(location 0 0.5)')]


dt = 0.05  # ms
T = 100  # ms

rec = recipe()
sim = A.simulation(rec)
sim.record(A.spike_recording.all)
sim.progress_banner()
sim.set_binning_policy(A.binning.regular, dt)
hdl = sim.sample((0, 0), # gid, off
                 A.regular_schedule(dt))

t0 = pc()
sim.run(100, 0.05)
t1 = pc()

print(sim.spikes())

fg, ax = plt.subplots()

for data, meta in sim.samples(hdl):
    print(meta)
    print(data)
    ax.plot(data[:, 0], data[:, 1])
    fg.savefig("single.pdf")
