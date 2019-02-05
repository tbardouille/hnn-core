# dipole.py - dipole-based analysis functions
#
# v 1.10.0-py35
# rev 2016-05-01 (SL: itertools and return data dir)
# last major: (SL: toward python3)
import numpy as np

from .paramrw import find_param
from .filt import hammfilt

# class Dipole() is for a single set of f_dpl and f_param


class Dipole():
    def __init__(self, f_dpl):  # fix to allow init from data in memory (not disk)
        """ some usage: dpl = Dipole(file_dipole, file_param)
            this gives dpl.t and dpl.dpl
        """
        self.units = None
        self.N = None
        self.__parse_f(f_dpl)

    # opens the file and sets units
    def __parse_f(self, f_dpl):
        x = np.loadtxt(open(f_dpl, 'r'))
        # better implemented as a dict
        self.t = x[:, 0]
        self.dpl = {
            'agg': x[:, 1],
            'L2': x[:, 2],
            'L5': x[:, 3],
        }
        self.N = self.dpl['agg'].shape[-1]
        # string that holds the units
        self.units = 'fAm'

    # truncate to a length and save here
    def truncate(self, t0, T):
        """ this is independent of the other stuff
            moved to an external function so as to not disturb the delicate genius of this object
        """
        self.t, self.dpl = self.truncate_ext(t0, T)

    # just return the values, do not modify the class internally
    def truncate_ext(self, t0, T):
        # only do this if the limits make sense
        if (t0 >= self.t[0]) & (T <= self.t[-1]):
            dpl_truncated = dict.fromkeys(self.dpl)
            # do this for each dpl
            for key in self.dpl.keys():
                dpl_truncated[key] = self.dpl[key][(
                    self.t >= t0) & (self.t <= T)]
            t_truncated = self.t[(self.t >= t0) & (self.t <= T)]
        return t_truncated, dpl_truncated

    # conversion from fAm to nAm
    def convert_fAm_to_nAm(self):
        """ must be run after baseline_renormalization()
        """
        for key in self.dpl.keys():
            self.dpl[key] *= 1e-6
        # change the units string
        self.units = 'nAm'

    def scale(self, fctr):
        for key in self.dpl.keys():
            self.dpl[key] *= fctr
        return fctr

    def smooth(self, winsz):
        if winsz <= 1:
            return
        #for key in self.dpl.keys(): self.dpl[key] = boxfilt(self.dpl[key],winsz)
        for key in self.dpl.keys():
            self.dpl[key] = hammfilt(self.dpl[key], winsz)

    # average stationary dipole over a time window
    def mean_stationary(self, opts_input={}):
        # opts is default AND input to below, can be modified by opts_input
        opts = {
            't0': 50.,
            'tstop': self.t[-1],
            'layer': 'agg',
        }
        # attempt to override the keys in opts
        for key in opts_input.keys():
            # check for each of the keys in opts
            if key in opts.keys():
                # special rule for tstop
                if key == 'tstop':
                    # if value in tstop is -1, then use end to T
                    if opts_input[key] == -1:
                        opts[key] = self.t[-1]
                else:
                    opts[key] = opts_input[key]
        # check for layer in keys
        if opts['layer'] in self.dpl.keys():
            # get the dipole that matches the xlim
            x_dpl = self.dpl[opts['layer']][(
                self.t > opts['t0']) & (self.t < opts['tstop'])]
            # directly return the average
            return np.mean(x_dpl, axis=0)
        else:
            print("Layer not found. Try one of %s" % self.dpl.keys())

    # finds the max value within a specified xlim
    # def max(self, layer, xlim):
    def lim(self, layer, xlim):
        # better implemented as a dict
        if layer is None:
            dpl_tmp = self.dpl['agg']
        elif layer in self.dpl.keys():
            dpl_tmp = self.dpl[layer]
        # set xmin and xmax
        if xlim is None:
            xmin = self.t[0]
            xmax = self.t[-1]
        else:
            xmin, xmax = xlim
            if xmin < 0.:
                xmin = 0.
            if xmax < 0.:
                xmax = self.f[-1]
        dpl_tmp = dpl_tmp[(self.t > xmin) & (self.t < xmax)]
        return (np.min(dpl_tmp), np.max(dpl_tmp))

    # simple layer-specific plot function
    def plot(self, ax, xlim, layer='agg'):
        # plot the whole thing and just change the xlim and the ylim
        # if layer is None:
        #     ax.plot(self.t, self.dpl['agg'])
        #     ymax = self.max(None, xlim)
        #     ylim = (-ymax, ymax)
        #     ax.set_ylim(ylim)
        if layer in self.dpl.keys():
            ax.plot(self.t, self.dpl[layer])
            ylim = self.lim(layer, xlim)
            # force ymax to be something sane
            # commenting this out for now, but
            # we can change if absolutely necessary.
            # ax.set_ylim(top=ymax*1.2)
            # set the lims here, as a default
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
        else:
            print("raise some error")
        return ax.get_xlim()

    # ext function to renormalize
    # this function changes in place but does NOT write the new values to the file
    def baseline_renormalize(self, f_param):
        # only baseline renormalize if the units are fAm
        if self.units == 'fAm':
            N_pyr_x = find_param(f_param, 'N_pyr_x')
            N_pyr_y = find_param(f_param, 'N_pyr_y')
            # N_pyr cells in grid. This is PER LAYER
            N_pyr = N_pyr_x * N_pyr_y
            # dipole offset calculation: increasing number of pyr cells (L2 and L5, simultaneously)
            # with no inputs resulted in an aggregate dipole over the interval [50., 1000.] ms that
            # eventually plateaus at -48 fAm. The range over this interval is something like 3 fAm
            # so the resultant correction is here, per dipole
            # dpl_offset = N_pyr * 50.207
            dpl_offset = {
                # these values will be subtracted
                'L2': N_pyr * 0.0443,
                'L5': N_pyr * -49.0502
                # 'L5': N_pyr * -48.3642,
                # will be calculated next, this is a placeholder
                # 'agg': None,
            }
            # L2 dipole offset can be roughly baseline shifted over the entire range of t
            self.dpl['L2'] -= dpl_offset['L2']
            # L5 dipole offset should be different for interval [50., 500.] and then it can be offset
            # slope (m) and intercept (b) params for L5 dipole offset
            # uncorrected for N_cells
            # these values were fit over the range [37., 750.)
            m = 3.4770508e-3
            b = -51.231085
            # these values were fit over the range [750., 5000]
            t1 = 750.
            m1 = 1.01e-4
            b1 = -48.412078
            # piecewise normalization
            self.dpl['L5'][self.t <= 37.] -= dpl_offset['L5']
            self.dpl['L5'][(self.t > 37.) & (self.t < t1)] -= N_pyr * \
                (m * self.t[(self.t > 37.) & (self.t < t1)] + b)
            self.dpl['L5'][self.t >= t1] -= N_pyr * \
                (m1 * self.t[self.t >= t1] + b1)
            # recalculate the aggregate dipole based on the baseline normalized ones
            self.dpl['agg'] = self.dpl['L2'] + self.dpl['L5']
        else:
            print("Warning, no dipole renormalization done because units were in %s" % (
                self.units))

    # function to write to a file!
    # f_dpl must be fully specified
    def write(self, f_dpl):
        with open(f_dpl, 'w') as f:
            for t, x_agg, x_L2, x_L5 in zip(self.t, self.dpl['agg'], self.dpl['L2'], self.dpl['L5']):
                f.write("%03.3f\t" % t)
                f.write("%5.4f\t" % x_agg)
                f.write("%5.4f\t" % x_L2)
                f.write("%5.4f\n" % x_L5)