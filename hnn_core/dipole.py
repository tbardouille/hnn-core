"""Class to handle the dipoles."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from numpy import convolve, hamming

from .parallel import _parallel_func


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


def _clone_and_simulate(params, trial_idx):
    from .network import Network

    if trial_idx != 0:
        params['prng_*'] = trial_idx

    net = Network(params, n_jobs=1)
    net.build()

    return _simulate_single_trial(net)


def _simulate_single_trial(net):
    """Simulate one trial."""
    from .parallel import rank, nhosts, pc, cvode
    from neuron import h
    h.load_file("stdrun.hoc")

    # Now let's simulate the dipole
    if rank == 0:
        print("running on %d cores" % nhosts)

    # global variables, should be node-independent
    h("dp_total_L2 = 0.")
    h("dp_total_L5 = 0.")

    # Set tstop before instantiating any classes
    h.tstop = net.params['tstop']
    h.dt = net.params['dt']  # simulation duration and time-step
    h.celsius = net.params['celsius']  # 37.0 - set temperature

    # We define the arrays (Vector in numpy) for recording the signals
    t_vec = h.Vector()
    t_vec.record(h._ref_t)  # time recording
    dp_rec_L2 = h.Vector()
    dp_rec_L2.record(h._ref_dp_total_L2)  # L2 dipole recording
    dp_rec_L5 = h.Vector()
    dp_rec_L5.record(h._ref_dp_total_L5)  # L5 dipole recording

    # sets the default max solver step in ms (purposefully large)
    pc.set_maxstep(10)

    # initialize cells to -65 mV, after all the NetCon
    # delays have been specified
    h.finitialize()

    def simulation_time():
        print('Simulation time: {0} ms...'.format(round(h.t, 2)))

    if rank == 0:
        for tt in range(0, int(h.tstop), 10):
            cvode.event(tt, simulation_time)

    h.fcurrent()
    # set state variables if they have been changed since h.finitialize
    h.frecord_init()
    # actual simulation - run the solver
    pc.psolve(h.tstop)

    pc.barrier()

    # these calls aggregate data across procs/nodes
    pc.allreduce(dp_rec_L2, 1)
    # combine dp_rec on every node, 1=add contributions together
    pc.allreduce(dp_rec_L5, 1)
    # aggregate the currents independently on each proc
    net.aggregate_currents()
    # combine net.current{} variables on each proc
    pc.allreduce(net.current['L5Pyr_soma'], 1)
    pc.allreduce(net.current['L2Pyr_soma'], 1)

    pc.barrier()  # get all nodes to this place before continuing

    dpl_data = np.c_[np.array(dp_rec_L2.to_python()) +
                     np.array(dp_rec_L5.to_python()),
                     np.array(dp_rec_L2.to_python()),
                     np.array(dp_rec_L5.to_python())]

    pc.gid_clear()
    pc.done()
    dpl = Dipole(np.array(t_vec.to_python()), dpl_data)
    if rank == 0:
        if net.params['save_dpl']:
            dpl.write('rawdpl.txt')

        dpl.baseline_renormalize(net.params)
        dpl.convert_fAm_to_nAm()
        dpl.scale(net.params['dipole_scalefctr'])
        dpl.smooth(net.params['dipole_smooth_win'] / h.dt)
    return dpl, net.spikes._times.to_python(), net.spikes._gids.to_python()


def simulate_dipole(net, n_trials=1, n_jobs=1):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    net : Network object
        The Network object specifying how cells are
        connected.
    n_trials : int
        The number of trials to simulate.
    n_jobs : int
        The number of jobs to run in parallel.

    Returns
    -------
    dpl: list | instance of Dipole
        The dipole object or list of dipole objects if n_trials > 1
    """
    parallel, myfunc = _parallel_func(_clone_and_simulate, n_jobs=n_jobs)
    out = parallel(myfunc(net.params, idx) for idx in range(n_trials))
    dpl, spike_times, spike_gids = zip(*out)
    net.spikes._times = list(spike_times)
    net.spikes._gids = list(spike_gids)
    net.spikes.update_types(net.gid_dict)
    return dpl


def read_dipole(fname, units='nAm'):
    """Read dipole values from a file and create a Dipole instance.

    Parameters
    ----------
    fname : str
        Full path to the input file (.txt)

    Returns
    -------
    dpl : Dipole
        The instance of Dipole class
    """
    dpl_data = np.loadtxt(fname, dtype=float)
    dpl = Dipole(dpl_data[:, 0], dpl_data[:, 1:4])
    if units == 'nAm':
        dpl.units = units
    return dpl


class Dipole(object):
    """Dipole class.

    Parameters
    ----------
    times : array (n_times,)
        The time vector
    data : array (n_times x 3)
        The data. The first column represents 'agg',
        the second 'L2' and the last one 'L5'

    Attributes
    ----------
    t : array
        The time vector
    dpl : dict of array
        The dipole with keys 'agg', 'L2' and 'L5'
    """

    def __init__(self, times, data):  # noqa: D102
        self.units = 'fAm'
        self.N = data.shape[0]
        self.t = times
        self.dpl = {'agg': data[:, 0], 'L2': data[:, 1], 'L5': data[:, 2]}

    def convert_fAm_to_nAm(self):
        """ must be run after baseline_renormalization()
        """
        for key in self.dpl.keys():
            self.dpl[key] *= 1e-6
        self.units = 'nAm'

    def scale(self, fctr):
        for key in self.dpl.keys():
            self.dpl[key] *= fctr
        return fctr

    def smooth(self, winsz):
        # XXX: add check to make sure self.t is
        # not smaller than winsz
        if winsz <= 1:
            return
        for key in self.dpl.keys():
            self.dpl[key] = _hammfilt(self.dpl[key], winsz)

    def plot(self, ax=None, layer='agg', show=True):
        """Simple layer-specific plot function.

        Parameters
        ----------
        ax : instance of matplotlib figure | None
            The matplotlib axis
        layer : str
            The layer to plot. Can be one of
            'agg', 'L2', and 'L5'
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if layer in self.dpl.keys():
            ax.plot(self.t, self.dpl[layer])
            ax.set_xlabel('Time (ms)')
            ax.set_title(layer)
        if show:
            plt.show()
        return ax.get_figure()

    def baseline_renormalize(self, params):
        """Only baseline renormalize if the units are fAm.

        Parameters
        ----------
        params : dict
            The parameters
        """
        if self.units != 'fAm':
            print("Warning, no dipole renormalization done because units"
                  " were in %s" % (self.units))
            return

        N_pyr_x = params['N_pyr_x']
        N_pyr_y = params['N_pyr_y']
        # N_pyr cells in grid. This is PER LAYER
        N_pyr = N_pyr_x * N_pyr_y
        # dipole offset calculation: increasing number of pyr
        # cells (L2 and L5, simultaneously)
        # with no inputs resulted in an aggregate dipole over the
        # interval [50., 1000.] ms that
        # eventually plateaus at -48 fAm. The range over this interval
        # is something like 3 fAm
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
        # L2 dipole offset can be roughly baseline shifted over
        # the entire range of t
        self.dpl['L2'] -= dpl_offset['L2']
        # L5 dipole offset should be different for interval [50., 500.]
        # and then it can be offset
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
        # recalculate the aggregate dipole based on the baseline
        # normalized ones
        self.dpl['agg'] = self.dpl['L2'] + self.dpl['L5']

    def write(self, fname):
        """Write dipole values to a file.

        Parameters
        ----------
        fname : str
            Full path to the output file (.txt)

        Outputs
        -------
        A tab separatd txt file where rows correspond
            to samples and columns correspond to
            1) time (s),
            2) aggregate current dipole (scaled nAm),
            3) L2/3 current dipole (scaled nAm), and
            4) L5 current dipole (scaled nAm)
        """
        X = np.r_[[self.t, self.dpl['agg'], self.dpl['L2'], self.dpl['L5']]].T
        np.savetxt(fname, X, fmt=['%3.3f', '%5.4f', '%5.4f', '%5.4f'],
                   delimiter='\t')
