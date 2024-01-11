#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
import pandas as pd
import numpy.random as rd
from scipy.stats import truncnorm
import arbor as A
from time import perf_counter as pc
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from arbor import units as U
except:
    # Units shim for arbor < 0.10.0
    class U:
        ms = 1.0
        mV = 1.0
        kHz = 1.0

dt = 0.05 * U.ms
T = 100 * U.ms

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


def make_hh(gid):
    # TODO figure out HH parameters
    # TODO figure out cell geometry
    tree = A.segment_tree()
    tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)
    center = "(location 0 0.5)"
    soma = "(tag 1)"
    decor = (
        A.decor()
        .set_property(Vm=-65 * U.mV)
        .paint(soma, A.density("hh"))
        .place(center, A.threshold_detector(-50 * U.mV), "source")
        .place(center, A.synapse("expsyn", {"tau": 0.5, "e": 0}), "synapse")
    )
    return A.cable_cell(tree, decor)


def make_spike_source(gid=0, *, tstart=0, tend=15, f=0.15):  # ms, ms, kHz
    return A.spike_source_cell(
        "source",
        A.poisson_schedule(
            tstart=tstart * U.ms, freq=f * U.kHz, tstop=tend * U.ms, seed=gid
        ),
    )


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


# Helper using the truncated normal distribution to avoid negative delays
def delay(mu, sigma, n):
    return truncnorm(
        (dt.value - mu) / sigma, (T.value - mu) / sigma, loc=mu, scale=sigma
    ).rvs(n)

class ucircuit(A.recipe):
    def __init__(
        self,
        *,
        l23=(0, 0),
        l4=(0, 0),
        l5=(0, 0),
        l6=(0, 0),
        nth=0,
        scale=1.0,
        w_scale=1.0,
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
        self.connection_probability = np.loadtxt("ucircuit/probabilities.csv", delimiter=",", dtype=float)
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
        self.weight_thalamic = 0  # 585.39
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

    def make_connection_parameters(self, src, tgt, n):
        # NOTE: The mean weight of the connection from L4E to L23E is doubled
        if src == ITH:
            w = np.ones(n) * self.weight_thalamic * self.weight_scale
            d = np.ones(n) * self.delay_thalamic
        elif src == I4E and tgt == I23E:
            w = rd.normal(2 * self.mean_weight_exc, self.stddev_weight_exc, n)
            d = delay(self.mean_delay_exc, self.stddev_delay_exc, n)
        elif src % 2 == 0:  # NOTE: all the excitatory ones are even.
            w = rd.normal(self.mean_weight_exc, self.stddev_weight_exc, n)
            d = delay(self.mean_delay_exc, self.stddev_delay_exc, n)
        else:
            w = rd.normal(self.mean_weight_inh, self.stddev_weight_inh, n)
            d = delay(self.mean_delay_inh, self.stddev_delay_inh, n)
        return w, d * U.ms

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
        return CELLS[pop](gid)

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
            p = self.connection_probability[tgt_pop][src_pop]
            n_src = self.size[src_pop]
            # Generate list of connection srcs
            srcs = np.argwhere(rd.random(n_src) < p)
            ws, ds = self.make_connection_parameters(src_pop, tgt_pop, srcs.size)
            # Now reify all those into connection objects
            # NOTE: We are simply skipping self connections here, but maybe
            # we need to re-draw those?
            old = len(res)
            res += [
                A.connection((src, "source"), "synapse", w, d)
                for (src, w, d) in zip(srcs, ws, ds)
                if src != tgt
            ]
            self.connections[src_pop][tgt_pop] += len(res) - old
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
                    self.weight_background * self.weight_scale,
                    A.poisson_schedule(tstart=0.0 * U.ms, freq=f * U.kHz, seed=gid),
                )
            ]


class parcellation(A.recipe):
    def __init__(self, num_tiles=1, scale=1.0):
        A.recipe.__init__(self)
        # NOTE: For now, all ucircuits are the same, but we can remedy that
        # by seeding the RNG with the tile number.
        self.ucircuit = ucircuit(
            l23=(20683, 5834),  # exc, inh
            l4=(21915, 5479),
            l5=(4850, 1065),
            l6=(14395, 2948),
            nth=902,
            scale=scale,
            w_scale=5e-6,
        )
        self.num_tiles = num_tiles
        # Tuning parameter.
        # NOTE: We assume here that
        # - connection strength is the sum over all relevant weights
        # - weights are normally distributed with some mu/sigma
        # - approx. str = count * mu
        self.mean_weight = 0.1
        self.stddev_weight = 0.05

        self.mean_delay = 0.75
        self.stddev_delay = 0.1

        # NOTE: Format: target column, source column?
        self.connection_probabilities = np.loadtxt(f'parcellations/{self.num_tiles}/weights.txt', dtype=float, delimiter=',')
        assert self.connection_probabilities.shape == (self.num_tiles, self.num_tiles)
        self.connection_lengths = np.loadtxt(f'parcellations/{self.num_tiles}/tract_lengths.txt', dtype=float, delimiter=',')
        assert self.connection_lengths.shape == (self.num_tiles, self.num_tiles)

        self.connections = np.zeros_like(self.connection_probabilities, dtype=int)

    def num_cells(self):
        return self.ucircuit.num_cells() * self.num_tiles

    def cell_kind(self, gid):
        return self.ucircuit.cell_kind(gid % self.ucircuit.num_cells())

    def cell_description(self, gid):
        return self.ucircuit.cell_description(gid % self.ucircuit.num_cells())

    def global_properties(self, kind):
        return self.ucircuit.global_properties(kind)

    def event_generators(self, gid):
        return self.ucircuit.event_generators(gid % self.ucircuit.num_cells())

    def connections_on(self, tgt):
        # Tile-internal GID
        loc = tgt % self.ucircuit.num_cells()

        res = self.ucircuit.connections_on(loc)
        tgt_pop = self.ucircuit.gid_to_pop(loc)
        tgt_tile = tgt // self.ucircuit.num_cells()
        # Inter-column connections go from L5 and L6 to L4
        if tgt_pop == I4E or tgt_pop == I4I:
            for src_tile in range(self.num_tiles):
                old = len(res)
                if src_tile == tgt_tile:
                    continue
                for src_pop in [I5E, I5I, I6E, I6I]:
                    p = self.connection_probabilities[tgt_tile][src_tile]
                    p = 0.01
                    n_src = self.ucircuit.size[src_pop]
                    # Generate list of connection srcs
                    srcs = np.argwhere(rd.random(n_src) < p)
                    ws = rd.normal(self.mean_weight, self.stddev_weight, srcs.size)
                    ds = delay(self.mean_delay, self.stddev_delay, srcs.size)
                    res += [
                        A.connection((src, "source"), "synapse", w, d * U.ms)
                        for (src, w, d) in zip(srcs, ws, ds)
                    ]
                new = len(res)
                self.connections[tgt_tile, src_tile] += new - old
        return res


rec = parcellation(num_tiles=68, scale=0.001)

ctx = A.context(threads=8)
sim = A.simulation(rec, ctx)
sim.record(A.spike_recording.all)
sim.progress_banner()

banner()
print(f"Set up the simulation, total cells N={rec.num_cells()} in n={rec.num_tiles} columns.")
print("\nuCircuit connections\n")

conn = pd.DataFrame(rec.ucircuit.connections)
conn.columns = LABELS
conn["TOTAL"] = conn.sum(axis=1)
conn = pd.concat(
    objs=[conn, pd.DataFrame(conn.sum(axis=0)).T], ignore_index=True, axis=0
)
conn.index = conn.columns

print(conn.to_string())

banner()
print("Column connections\n")

print("       ", end='')
for n in range(rec.num_tiles):
    print(f"{n:>6d}", end=' ')
print()
for n in range(rec.num_tiles):
    print(f"{n:>6d}", end=' ')
    for m in range(rec.num_tiles):
        print(f"{rec.connections[n, m]:>6d}", end=' ')
    print()

banner()
print(f"Running simulation for T={T} at dt={dt}")

t0 = pc()
sim.run(T, dt)
t1 = pc()
print(f"Done, took {t1 - t0:0.3f}s.")

banner()
print("Spikes\n")

gs, ls, ts, ps, cs = [], [], [], [], []
events = [[] for _ in range(rec.num_cells())]
for (gid, lid), t in sim.spikes():
    pop = rec.ucircuit.gid_to_pop(gid % rec.ucircuit.num_cells())
    ps.append(LABELS[pop])
    gs.append(gid)
    ls.append(lid)
    ts.append(t)
    cs.append(gid // rec.ucircuit.num_cells())
    events[gid].append(t)

colors = [sns.color_palette()[rec.ucircuit.gid_to_pop(gid % rec.ucircuit.num_cells())]
          for gid in range(rec.num_cells())]

spikes = pd.DataFrame({"time": ts, "lid": ls, "gid": gs, "pop": ps, "col": cs})
counts = spikes.groupby(["pop", "col"]).count().time.unstack(1)
totals = counts.sum(axis=0)
counts.loc['TOTAL'] = totals.values
totals = counts.sum(axis=1)
counts.loc[:, 'TOTAL'] = totals.values
print(counts.to_string())

fg, ax = plt.subplots()

ax.eventplot(events, colors=colors)
ax.set_xlabel("Time $(t/ms)$")
ax.set_ylabel("GID")
ax.set_ylim(0, rec.ucircuit.N)
ax.set_xlim(0, T.value)
fg.savefig("main-spikes.pdf")
fg.savefig("main-spikes.png")
