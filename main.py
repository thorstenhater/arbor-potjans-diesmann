#!/usr/bin/env python3

import arbor as A

class recipe(A.recipe):
    def __init__(self, N):
        A.recipe.__init__(self)
        self.N = N

    def num_cells(self):
        return self.N

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        tree = A.segment_tree()
        tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)

        decor = (
            A.decor()
            .set_property(Vm=-40)
            .paint('(tag 1)', A.density("hh"))
            .place('(location 0 0.5)', A.threshold_detector(-10), "detector")
            .place('(location 0 0.75)', A.synapse("expsyn"), "synapse")
        )

        return A.cable_cell(tree, decor)

    def global_properties(self, kind):
        return A.neuron_cable_properties()

    def connections_on(self, gid):
        src = (gid - 1) % self.N
        return [A.connection((src, "detector"), # source spec = gid, label
                             "synapse",         # target
                             0.01,              # weight
                             5)]                # axonal delay

    def event_generators(self, gid):
        if gid == 0:
            return [A.event_generator("synapse",
                                      0.05,
                                      A.regular_schedule(10,
                                                         1,
                                                         20))]
        else:
            return []

A.mpi_init()
com = A.mpi_comm()
ctx = A.context(mpi=com)

rec = recipe(16)
sim = A.simulation(rec, ctx)

sim.record(A.spike_recording.all)

sim.run(100, 0.05)

for (gid, lid), t in sim.spikes():
    print(f" * {t:.3f} gid={gid:3d} lid={lid:3d}")
