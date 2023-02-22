#!/usr/bin/env python3

import arbor as A
from time import perf_counter as pc
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
        .paint(soma, A.density("hh", {}))
        .place(center, A.threshold_detector(-50), "source")
        .place(center, A.synapse("expsyn", {"tau": 0.5, "e": 0}), "synapse")
    )
    return A.cable_cell(tree, decor)


class recipe(A.recipe):
    def __init__(self, scale=5e-7):
        A.recipe.__init__(self)
        # NOTE original background frequency times the indegree
        self.f_background = 8e-3 * 1600  # kHz
        # NOTE We need to scale down the weight to allow the HH mechanism to recover
        self.weight_background = 585.39 * scale
        # NOTE original background frequency times the indegree
        self.f_thalamic = 15e-3 * 902 * 0.0983
        # NOTE We need to scale down the weight to allow the HH mechanism to recover
        self.weight_thalamic = 585.39 * scale

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return make_hh()

    def global_properties(self, kind):
        return A.neuron_cable_properties()

    def event_generators(self, gid):
        return [
            A.event_generator(
                "synapse",
                self.weight_background,
                A.poisson_schedule(tstart=0.0, freq=self.f_background),
            ),
            A.event_generator(
                "synapse",
                self.weight_thalamic,
                A.poisson_schedule(tstart=0.0, freq=self.f_thalamic),
            ),
        ]

    def probes(self, gid):
        return [A.cable_probe_membrane_voltage("(location 0 0.5)")]


dt = 0.05  # ms
T = 100  # ms

rec = recipe()
sim = A.simulation(rec)
sim.record(A.spike_recording.all)
sim.progress_banner()
sim.set_binning_policy(A.binning.regular, dt)
hdl = sim.sample((0, 0), A.regular_schedule(dt))  # gid, off

t0 = pc()
sim.run(100, 0.05)
t1 = pc()

print(sim.spikes())

fg, ax = plt.subplots()

for data, meta in sim.samples(hdl):
    ax.plot(data[:, 0], data[:, 1])
    fg.savefig("single.pdf")
    fg.savefig("single.png")
