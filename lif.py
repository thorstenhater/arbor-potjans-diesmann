#!/usr/bin/env python3

import arbor as A
from time import perf_counter as pc
import matplotlib.pyplot as plt


def make_iaf():
    res = A.lif_cell("source", "synapse")
    res.tau_m = 10
    res.t_ref = 2
    res.C_m = 250
    res.E_L = -65
    res.V_m = -65
    res.E_R = -65
    return res


class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.f_background = 8e-3  # kHz
        self.k_background = 2100
        self.weight_background = 585.39
        # Thalamic inputs
        self.f_thalamic = 15e-3  # kHz
        self.weight_thalamic = 585.39
        self.delay_thalamic = 1.5

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def cell_description(self, gid):
        return make_iaf()

    def global_properties(self, kind):
        return None

    def event_generators(self, gid):
        return [
            # thalamic input for L4e
            A.event_generator(
                "synapse",
                self.weight_thalamic,
                A.poisson_schedule(
                    tstart=20, tstop=40, freq=self.f_thalamic * 902 * 21915 * 0.982
                ),
            ),
            A.event_generator(
                "synapse",
                self.weight_background,
                A.poisson_schedule(
                    tstart=0.0, freq=self.f_background * self.k_background
                ),
            ),
        ]

    def probes(self, gid):
        return [A.lif_probe_voltage()]


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
    print(meta)
    print(data)
    ax.plot(data[:, 0], data[:, 1])
    fg.savefig("lif.pdf")
