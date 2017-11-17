import numpy as np
import os
import sys
import itertools as itl

# CONTINUE by opening existing configuration files from the first pass and altering them with the second parameter vals

class ShuffleConfig:

    def __init__(self, ini_file, limits_file):

        self.cf = ini_file
        self.lf = limits_file

        # build parameter dictionary containing limits from file
        self.params = self.read_limits()

        # add all possible combinations to parameter dictionary
        self.get_range()

        # read in config file template
        self.c = self.read_conf()

        # write config files with alternate settings
        self.make_configs()

    def make_configs(self):
        """
        Build a config file for each unique setting.
        """

        step = 0

        # process first level of params
        lev = []

        for index, k in enumerate(self.params.keys()):

            # create output file name
            bf = os.path.join(os.path.dirname(self.cf), 'shuffle_config_{0}'.format(index))

            # if first pass use template
            if index == 0:
                # split string to isolate parameter and value
                a = self.c.split(k)

            # otherwise alter first pass files

            b = a[1].split()

            # get values to insert
            vals = self.params[k][1]

            # create a file for each configuration
            for idx, v in enumerate(vals):

                # add iter to file string
                f = '{0}_{1}.csv'.format(bf, idx)

                # build replacement string for parmeter=value
                s = '{0}={1}'.format(k, v)

                # replace config string with new param string
                b[0] = s

                # join list by newline
                n = '\n'.join(b)

                # write out file
                with open(f, 'w') as out:

                    for line in n:

                        out.write(line)

            # remove key from dictionary
            self.params.pop(k, None)

    def read_conf(self):
        """
        Read in config file template.
        :return:  File as string
        """
        with open(self.cf, 'rU') as get:
            os = ''
            for i in get:
                # remove comment lines
                if i[0] == '#':
                    continue
                else:
                    os += i.replace(' ', '')

            return os

    def get_range(self):

        for k in self.params.keys():

            # unpack values
            v = self.params[k][0]

            # add all possible combinations to dict
            self.params[k].append(np.arange(v[0], v[1] + v[2], v[2]))

    def read_limits(self):
        """
        Read limits of CSV file containing a header.  Fields to be parameter, min_val, max_val, increment
        :return:
        """
        d = {}
        with open(self.lf, 'rU') as get:
            for idx, i in enumerate(get):

                # pass header
                if idx > 0:
                    si = i.strip().split(',')

                    # detect duplicate parameters in limits file
                    if si[0] in d:
                        print("ERROR - Duplicate parameter detected: {0}\n".format(si[0]))
                        print("Fix error in limits CSV file and rerun.\n")
                        print("Exiting...\n")
                        sys.exit()

                    else:
                        d[si[0]] = [[float(si[1]), float(si[2]), float(si[3])]]

        return d


def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def priority_allocation(f, n=1):
    """
    Create a numpy array to hold random permutations of possible settings for
    priority allocation.

    :param f:   input file path and name with extension
    :param n:   number of permutations
    :return:    numpy array shape [n, num_fields, num_fields]
    """

    with open(f, 'rU') as get:

        # create a list of all categories in header
        h = get.next().strip().split(',')[1:]

        # get a list of possible values as int
        v = [int(i) for i in get.next().strip().split(',')[1:]]

        # build array of unique permutations of values
        p = list(set(itl.permutations(v)))

        # build arrays to hold permutations
        arr = np.zeros(shape=(n, len(h), len(h)))

        for idx, c in enumerate(h):

            # get index locations of each category; these are where the categories value equals 0
            c_idx = [ix for ix, i in enumerate(p) if i[idx] == 0]

            for step in range(n):

                go = True

                while go:

                    # get random selection of category index
                    rx = np.random.choice(c_idx)

                    # add the selections array to the list if it does not already exists
                    if (arr[step, idx, :] == p[rx]).all():
                        go = True
                    else:
                        arr[step, idx, :] = p[rx]
                        go = False
    return arr


def treatment_order(f):
    """
    Get every possible combination of treatment order configurations.
    :param f:               Treatment order file path
    :return:                A list of tuples containing treatment order configurations.
    """
    c = []
    v = []

    with open(f, 'rU') as get:

        # pass header
        get.next()

        for i in get:

            s = i.strip().split(',')

            c.append(s[0])
            v.append(int(s[1]))

    # build array of unique permutations of values
    p = list(set(itl.permutations(v)))

    return p


def scenarios(lim):
    """
    Create all possible values in one dimensional settings for parameters defined in the limits file.
    :param lim              Input CSV with header containing limits:  param,min_val,max_val,interval
    :return:                Dictionary where parameter: array([n1, ...])
    """
    d = {}
    with open(lim, 'rU') as get:

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
