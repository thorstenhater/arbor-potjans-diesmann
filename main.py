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


dt = 0.05  # ms
T = 100  # ms
VERBOSE = False


def banner():
    print()
    print("-" * 80)
    print()


def make_iaf():
    return A.lif_cell(
        source="source",
        target="synapse",
        tau_m=10,
        t_ref=2,
        C_m=250,
        E_L=-65,  # TODO
        V_m=-65,  # TODO
        E_R=-65,
    )


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
        .paint(soma, A.density("hh"))
        .place(center, A.threshold_detector(-50), "source")
        .place(center, A.synapse("expsyn", {"tau": 0.5, "e": 0}), "synapse")
    )
    return A.cable_cell(tree, decor)


def make_spike_source(*, tstart=0, tend=15, f=0.15):  # ms, ms, kHz
    return A.spike_source_cell("source", A.poisson_schedule(tstart, f, tend))


make_l23e = make_hh
make_l23i = make_hh
make_l4e = make_hh
make_l4i = make_hh
make_l5e = make_hh
make_l5i = make_hh
make_l6e = make_hh
make_l6i = make_hh
make_th = make_spike_source

POPS = range(9)
(
    I23E,
    I23I,
    I4E,
    I4I,
    I5E,
    I5I,
    I6E,
    I6I,
    ITH,
) = POPS

CELLS = [
    make_l23e,
    make_l23i,
    make_l4e,
    make_l4i,
    make_l5e,
    make_l5i,
    make_l6e,
    make_l6i,
    make_th,
]

LABELS = [
    f"{l}-{t}"
    for l in [
        23,
        4,
        5,
        6,
    ]
    for t in "ei"
] + ["th"]


class recipe(A.recipe):
    def __init__(
        self,
        *,
        l23=(0, 0),
        l4=(0, 0),
        l5=(0, 0),
        l6=(0, 0),
        nth=0,
        scale=1.0,
        w_scale=1.0
    ):
        A.recipe.__init__(self)
        # Sizes of sub-populations
        n23e, n23i = l23
        n4e, n4i = l4
        n5e, n5i = l5
        n6e, n6i = l6
        size = [n23e, n23i, n4e, n4i, n5e, n5i, n6e, n6i, nth]
        self.size = np.array([int(n * scale) for n in size])
        # Offset of population I into the gids **AND** one past last pop
        self.offset = np.cumsum(np.insert(self.size, 0, 0))
        # total size
        self.N = self.offset[-1]
        # Probability to connect between a target population and a source population.
        # Layout: [tgt][src]
        self.connection_probability = np.array(
            [
                [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0, 0.0],
                [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0, 0.0],
                [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0, 0.0983],
                [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0, 0.0619],
                [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0, 0.0],
                [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0, 0.0],
                [
                    0.0156,
                    0.0066,
                    0.0211,
                    0.0166,
                    0.0572,
                    0.0197,
                    0.0396,
                    0.2252,
                    0.0512,
                ],
                [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443, 0.0196],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )
        # Scale weights for HH ./. LIF
        self.weight_scale = w_scale
        # Delays
        self.mean_delay_exc = 1.5
        self.mean_delay_inh = 0.5 * self.mean_delay_exc
        self.stddev_delay_exc = 0.5 * self.mean_delay_exc
        self.stddev_delay_inh = 0.5 * self.mean_delay_inh
        # Weights
        self.mean_weight_exc = 585.39
        self.mean_weight_inh = -4 * self.mean_weight_exc
        self.stddev_weight_exc = 0.1 * self.mean_weight_exc
        self.stddev_weight_inh = 0.4 * self.mean_weight_exc
        # Background
        self.f_background = 8e-3  # kHz
        # Indegree of background connection, used as a scale for the frequency here
        # TODO test/check if this holds water
        self.k_background = np.array(
            [1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100, 0]
        )
        self.weight_background = 585.39
        # Thalamic inputs
        self.f_thalamic = 15e-3
        self.weight_thalamic = 585.39
        self.delay_thalamic = 1.5
        # Record synapse counts for reporting. We'd expect p_s_t*n_s*n_t on
        # average for source and target populations.
        #
        # NOTE: We could also pregenerate all connections here, but that seems
        # problematic in the face of a target of 300M synapses.
        #
        # NOTE: This will also fall flat when multi-threading and/or MPI is
        # used.
        self.connections = defaultdict(lambda: np.zeros_like(POPS))

    def make_connection(self, src, tgt):
        # NOTE: The mean weight of the connection from L4E to L23E is doubled
        if src == ITH:
            w = self.weight_thalamic*self.weight_scale
            d = self.delay_thalamic
        elif src == I4E and tgt == I23E:
            w = rd.normal(2 * self.mean_weight_exc, self.stddev_weight_exc)
            d = rd.normal(self.mean_delay_exc, self.stddev_delay_exc)
        elif src % 2 == 0:  # NOTE: all the excitatory ones are even.
            w = rd.normal(self.mean_weight_exc, self.stddev_weight_exc)
            d = rd.normal(self.mean_delay_exc, self.stddev_delay_exc)
        else:
            w = rd.normal(self.mean_weight_inh, self.stddev_weight_inh)
            d = rd.normal(self.mean_delay_inh, self.stddev_delay_inh)
        # NOTE: There's a bug on clang (at least on MacOS) that results in
        # broken simulations if d < dt, so fix it here. Usually Arbor should do
        # this on its own.
        if d < dt:
            d = dt
            warning(
                f"Connection {src} -> {tgt} has delay less than dt={dt}, using dt instead."
            )
        return w, d

    def gid_to_pop(self, gid):
        # return the first IDX where our GID is less than POP[IDX+1]
        for off, idx in zip(self.offset[1:], POPS):
            if gid < off:
                return idx
        if gid >= self.N:
            raise RuntimeError(f"GID {gid} out-of-bounds.")
        raise RuntimeError("Never!")

    def num_cells(self):
        return self.N

    def cell_kind(self, gid):
        pop = self.gid_to_pop(gid)
        if pop == ITH:
            return A.cell_kind.spike_source
        else:
            return A.cell_kind.cable

    def cell_description(self, gid):
        pop = self.gid_to_pop(gid)
        return CELLS[pop]()

    def global_properties(self, kind):
        if kind == A.cell_kind.cable:
            return A.neuron_cable_properties()
        else:
            return None

    def connections_on(self, tgt):
        res = []
        tgt_pop = self.gid_to_pop(tgt)
        # Scan all Population types
        for src_pop in POPS:
            n = 0
            p = self.connection_probability[tgt_pop][src_pop]
            n_src = self.size[src_pop]
            # Generate list of connection srcs
            srcs = np.argwhere(rd.random(n_src) < p)
            # Now reify all those into connection objects
            # NOTE: We are simply skipping self connections here, but maybe
            # we need to re-draw those?
            for src in srcs:
                if src == tgt:
                    continue
                w, d = self.make_connection(src_pop, tgt_pop)
                res.append(A.connection((src, "source"), "synapse", w, d))
                n += 1
            self.connections[src_pop][tgt_pop] += n
        return res

    def event_generators(self, gid):
        pop = self.gid_to_pop(gid)
        if pop == ITH:
            return []
        else:
            # Model the background
            f = self.f_background * self.k_background[pop]
            return [
                A.event_generator(
                    "synapse",
                    self.weight_background*self.weight_scale,
                    A.poisson_schedule(tstart=0.0, freq=f),
                )
            ]


rec = recipe(
    l23=(20683, 5834),  # exc, inh
    l4=(21915, 5479),
    l5=(4850, 1065),
    l6=(14395, 2948),
    nth=902,
    scale=0.1,
    w_scale=7.5e-7
)

ctx = A.context(threads=8)
sim = A.simulation(rec, ctx)
sim.record(A.spike_recording.all)
sim.progress_banner()
sim.set_binning_policy(A.binning.regular, dt)

banner()
print(f"Set up the simulation, total cells N={rec.N}")
print("\nConnections\n")

conn = pd.DataFrame(rec.connections)
conn.columns = LABELS
conn["TOTAL"] = conn.sum(axis=1)
conn = pd.concat(objs=[conn, pd.DataFrame(conn.sum(axis=0)).T], ignore_index=True, axis=0)
conn.index = conn.columns

print(conn.to_string())

banner()
print(f"Running simulation for {T}ms at dt={dt}ms")

t0 = pc()
sim.run(T, dt)
t1 = pc()
print(f"Done, took {t1 - t0:0.3f}s.")

banner()
print("Spikes\n")

gs, ls, ts, ps = [], [], [], []
events = [[] for _ in range(rec.N)]
for (gid, lid), t in sim.spikes():
    pop = rec.gid_to_pop(gid)
    ps.append(LABELS[pop])
    gs.append(gid)
    ls.append(lid)
    ts.append(t)
    events[gid].append(t)

colors = [sns.color_palette()[rec.gid_to_pop(gid)] for gid in range(rec.N)]

spikes = pd.DataFrame({"time": ts, "lid": ls, "gid": gs, "pop": ps})
counts = pd.DataFrame(spikes.groupby("pop").count()["time"])
counts.columns = ["count"]
counts = pd.concat(
    objs=[counts, pd.DataFrame({"count": counts["count"].sum()}, index=["TOTAL"])]
)
print(counts.to_string())

fg, ax = plt.subplots()

ax.eventplot(events, colors=colors)
ax.set_xlabel('Time $(t/ms)$')
ax.set_ylabel('GID')
ax.set_ylim(0, rec.N)
ax.set_xlim(0, T)
fg.savefig("main-spikes.pdf")
fg.savefig("main-spikes.png")
