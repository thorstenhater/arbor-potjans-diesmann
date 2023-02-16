#!/usr/bin/env python3

import numpy as np
import numpy.random as rd
import arbor as A

def make_l23i():
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


def make_l23e():
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


def make_l4i():
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


def make_l4e():
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


def make_l5i():
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


def make_l5e():
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

def make_l6i():
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


def make_l6e():
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


POPS = range(8)
I23E, I23I, I4E, I4I, I5E, I5I, I6E, I6I, = POPS

CELLS = [make_l23e,
         make_l23i,
         make_l4e,
         make_l4i,
         make_l5e,
         make_l5i,
         make_l6e,
         make_l6i,]

class recipe(A.recipe):
    def __init__(self, *, l23=(0,0), l4=(0,0), l5=(0,0), l6=(0,0)):
        A.recipe.__init__(self)
        # Sizes of sub-populations
        n23e, n23i = l23
        n4e, n4i = l4
        n5e, n5i = l5
        n6e, n6i = l6
        self.sizes = np.array([n23e, n23i, n4e, n4i, n5e, n5i, n6e, n6i])
        # Offset of population I into the gids **AND** one past last pop
        self.offset = np.cumsum(np.insert(self.size, 0, 0))
        # total size
        self.N = self.offset[-1]
        # Probability to connect between a target population and a source population.
        # Layout: [tgt][src]
        self.connection_probability = np.array([[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.],
                                                [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.],
                                                [0.0077, 0.0059, 0.0497, 0.135, 0.0067,  0.0003, 0.0453, 0.],
                                                [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.],
                                                [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
                                                [0.0548, 0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.],
                                                [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                                                [0.0364, 0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]])

        # Delays
        self.mean_delay_exc    =  1.5
        self.mean_delay_inh    =  0.5*self.mean_delay_exc
        self.stddev_delay_exc  =  0.5*self.mean_delay_exc
        self.stddev_delay_inh  =  0.5*self.mean_delay_inh
        # Weights
        self.mean_weight_exc   =  585.39
        self.mean_weight_inh   =  -4*self.mean_weight_exc
        self.stddev_weight_exc =  0.1*self.mean_weight_exc
        self.stddev_weight_inh =  0.4*self.mean_weight_exc

    def make_connection(self, src, tgt):
        # NOTE: The mean weight of the connection from L4E to L23E is doubled
        if src == I4E and tgt == I23E:
            w = rd.normal(2*self.mean_weight_exc, self.stddev_weight_exc)
            d = rd.normal(self.mean_delay_exc, self.stddev_delay_exc)
        elif src % 2 == 0: # NOTE: all the excitatory ones are even.
            w = rd.normal(self.mean_weight_exc, self.stddev_weight_exc)
            d = rd.normal(self.mean_delay_exc, self.stddev_delay_exc)
        else:
            w = rd.normal(self.mean_weight_inh, self.stddev_weight_inh)
            d = rd.normal(self.mean_delay_inh, self.stddev_delay_inh)
        return w, d

    def gid_to_pop(self, gid):
        if gid >= self.offset[I23E]:
            return I23E
        elif gid >= self.offset[I23I]:
            return I23I
        elif gid >= self.offset[I4E]:
            return I4E
        elif gid >= self.offset[I4I]:
            return I4I
        elif gid >= self.offset[I5E]:
            return I5E
        elif gid >= self.offset[I5I]:
            return I5I
        elif gid >= self.offset[I6E]:
            return I6E
        elif gid >= self.offset[I6I]:
            return I6I
        elif gid >= self.N:
            raise RuntimeError(f"GID {gid} out-of-bounds.")
        raise RuntimeError("Never!")

    def num_cells(self):
        return self.N

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        idx = self.gid_to_pop(gid)
        return CELLS[idx]()

    def global_properties(self, kind):
        return A.neuron_cable_properties()

    def connections_on(self, tgt):
        res = []
        tgt_pop = self.gid_to_pop(tgt)

        # Scan all Population types
        for src_pop in POPS:
            p = self.connection_probability[tgt_pop][src_pop]
            n_src = self.sizes[src_pop]
            # Generate list of connection srcs
            srcs = np.where(rd.random(n_src) < p)
            # Now reify all those into connection objects
            # NOTE: We are simpliy skipping self connections here, but maybe
            # we need to re-draw those?
            for src in srcs:
                if src == tgt:
                    continue
                w, d = self.make_connection(src_pop, tgt_pop)
                res.append(A.connection((src, "detector"), "synapse", w, d))
        return res

rec = recipe(4, 4, 4, 4, 4, 4, 4, 4)
sim = A.simulation(rec)
sim.run(100, 0.05)
