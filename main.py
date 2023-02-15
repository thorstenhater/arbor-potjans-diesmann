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


I23E, I23I, I4E, I4I, I5E, I5I, I6E, I6I, = range(8)


class recipe(A.recipe):
    def __init__(self, n23e, n23i, n4e, n4i, n5e, n5i, n6e, n6i, ):
        A.recipe.__init__(self)
        self.n23e, self.n23i, self.n4e, self.n4i, self.n5e, self.n5i, self.n6e, self.n6i = n23e, n23i, n4e, n4i, n5e, n5i, n6e, n6i,
        self.sizes = np.array([n23e, n23i, n4e, n4i, n5e, n5i, n6e, n6i])
        self.offset = np.cumsum(self.sizes)
        self.N = np.sum(self.sizes)
        # [tgt][src]
        self.connection_probability = np.array([[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.],
                                                [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.],
                                                [0.0077, 0.0059, 0.0497, 0.135, 0.0067,  0.0003, 0.0453, 0.],
                                                [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.],
                                                [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
                                                [0.0548, 0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.],
                                                [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                                                [0.0364, 0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]])
        self.mean_delay_exc    =  1.5
        self.mean_delay_inh    =  0.75
        self.stddev_delay_exc  =  0.1  # ??
        self.stddev_delay_inh  = -0.4  # ??
        self.mean_weight_exc   =  1.5  # ??
        self.mean_weight_inh   =  0.75 # ??
        self.stddev_weight_exc =  0.1
        self.stddev_weight_inh = -0.4


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
        else:

    def num_cells(self):
        return self.N

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        idx = self.gid_to_pop(gid)
        fun = [make_l23e,
               make_l23i,
               make_l4e,
               make_l4i,
               make_l5e,
               make_l5i,
               make_l6e,
               make_l6i,]
        return fun[idx]()

    def global_properties(self, kind):
        return A.neuron_cable_properties()

    def connections_on(self, gid):
        res = []
        tgt_pop = self.gid_to_pop(gid)

        for src_pop in range(8):
            p = self.connection_probability[tgt_pop][src_pop]
            n_src = self.sizes[src_pop]
            ds = rd.random(n_src) < p # [0, 0, 1, 1, 0, 1, ...]
            for src in range(n_src):
                if ds[src]:
                    if src_pop % 2 == 0: # all the excitatory ones are even.
                        w = rd.normal(self.mean_weight_exc, self.stddev_weight_exc)
                        d = rd.normal(self.mean_delay_exc, self.stddev_delay_exc)
                    else:
                        w = rd.normal(self.mean_weight_inh, self.stddev_weight_inh)
                        d = rd.normal(self.mean_delay_inh, self.stddev_delay_inh)

                    res.append(A.connection((src, "detector"), # source spec = gid, label
                                            "synapse",         # target
                                            w,                 # weight
                                            d))                # delay
        return res

rec = recipe(4, 4, 4, 4, 4, 4, 4, 4)
sim = A.simulation(rec)
sim.run(100, 0.05)
