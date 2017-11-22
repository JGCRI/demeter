"""
Demeter example run.

Copyright (c) 2017, Battelle Memorial Institute

Open source under license BSD 2-Clause - see LICENSE and DISCLAIMER

@author:  Chris R. Vernon (chris.vernon@pnnl.gov)
"""
import numpy as np
import os
import itertools as itl


class RandomConfig:

    def __init__(self, c):

        self.perms = c.permutations
        self.priority_file = c.priority_allocation
        self.treat_file = c.treatment_order
        self.limits_file = os.path.join(c.limits_file)

        # get all possible combinations from limits file
        d = self.scenarios()

        self.r_intense_ratio = d['intensification_ratio']
        self.r_select_thresh = d['selection_threshold']
        self.r_kernel = d['kerneldistance']

        self.r_priority = None
        self.r_treatment_order = None

        # populate priority and treatement order possibilities
        self.priority_allocation()
        self.treatment_order()

        # get a list of configurations
        self.mix = []
        self.selection()

    def priority_allocation(self):
        """
        Create a numpy array to hold random permutations of possible settings for
        priority allocation.

        :param f:   input file path and name with extension
        :param n:   number of permutations
        :return:    numpy array shape [n, num_fields, num_fields]
        """

        with open(self.priority_file, 'rU') as get:

            # create a list of all categories in header
            h = get.next().strip().split(',')[1:]

            # get a list of possible values as int
            v = [int(i) for i in get.next().strip().split(',')[1:]]

            # build array of unique permutations of values
            p = list(set(itl.permutations(v)))

            # build arrays to hold permutations
            self.r_priority = np.zeros(shape=(self.perms, len(h), len(h)))

            for idx, c in enumerate(h):

                # get index locations of each category; these are where the categories value equals 0
                c_idx = [ix for ix, i in enumerate(p) if i[idx] == 0]

                for step in range(self.perms):

                    go = True

                    while go:

                        # get random selection of category index
                        rx = np.random.choice(c_idx)

                        # add the selections array to the list if it does not already exists
                        if (self.r_priority[step, idx, :] == p[rx]).all():
                            go = True
                        else:
                            self.r_priority[step, idx, :] = p[rx]
                            go = False

    def treatment_order(self):
        """
        Get every possible combination of treatment order configurations.
        :param f:               Treatment order file path
        :return:                A list of tuples containing treatment order configurations.
        """
        v = []
        with open(self.treat_file, 'rU') as get:

            # pass header
            get.next()

            for i in get:

                s = i.strip().split(',')
                v.append(int(s[1]))

        # build array of unique permutations of values
        self.r_treatment_order = list(set(itl.permutations(v)))

    def scenarios(self):
        """
        Create all possible values in one dimensional settings for parameters defined in the limits file.
        :param lim              Input CSV with header containing limits:  param,min_val,max_val,interval
        :return:                Dictionary where parameter: array([n1, ...])
        """
        d = {}
        with open(self.limits_file, 'rU') as get:

            # pass header
            get.next()

            for idx, i in enumerate(get):

                # split row
                r = i.strip().split(',')

                mn = float(r[1])
                mx = float(r[2])
                st = float(r[3])

                if st >= 1:
                    mx += st

                d[r[0]] = np.arange(mn, mx, st)

        return d

    def selection(self):
        """
        Build a possible number of configurations for parameters.
        :return:
        """

        for i in range(self.perms):

            # get priority order; number of these is already randomly chosen when created; there will not be duplicates
            pri = self.r_priority[i]

            # get treatment order by random choice
            trt = self.r_treatment_order[np.random.choice(len(self.r_treatment_order))]

            # get random intensification ratio
            iv = self.r_intense_ratio[np.random.choice(len(self.r_intense_ratio))]

            # get random selection threshold value
            st = self.r_select_thresh[np.random.choice(len(self.r_select_thresh))]

            # get random kernel distance value
            kd = int(self.r_kernel[np.random.choice(len(self.r_kernel))])

            # scenario suffix so jobs will not overwrite each other
            suf = i

            # add to output object
            self.mix.append([pri, trt, iv, st, kd, suf])
