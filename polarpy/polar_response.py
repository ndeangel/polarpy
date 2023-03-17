import h5py
import numba as nb
import numpy as np
from interpolation.splines import eval_linear
from tqdm import tqdm

#import scipy.interpolate as interpolate


energy_resolved = "pdheavyside" #None

if not energy_resolved:  # 3 parameters
    spec = [
        ("_values", nb.float64[:, :, :]),
        (
            "_grid",
            nb.typeof(
                (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64),)
            ),
        ),
    ]

elif energy_resolved == "pdheavyside":  # 5 parameters
    spec = [
        ("_values", nb.float64[:, :, :, :, :]),
        (
            "_grid",
            nb.typeof(
                (np.zeros(5, dtype=np.float64), np.zeros(5, dtype=np.float64), np.zeros(5, dtype=np.float64), np.zeros(5, dtype=np.float64), np.zeros(5, dtype=np.float64),)
            ),
        ),
    ]

@nb.jitclass(spec)
class FastGridInterpolate(object):

    def __init__(self, grid, values):
        self._grid = grid
        self._values = np.ascontiguousarray(values)

    def evaluate(self, v):

        return eval_linear(self._grid, self._values, v)


class PolarResponse(object):

    def __init__(self, response_file):
        """
        Construct the polar response from the HDF5 response file.
        This is the POLARIZATION response.


        :param response_file: 
        :returns: 
        :rtype: 

        """

        self._rsp_file = response_file

        # pre interpolate the response for fitting

        self._interpolate_rsp()

    def _interpolate_rsp(self):
        """
        Builds the interpolator for the response. This is currently incredibly slow
        and should be improved

        """

        # now go through the response and extract things

        with h5py.File(self._rsp_file, 'r') as f:

            energy = f['energy'][()]

            ene_lo, ene_hi = [], []

            # the bin widths are hard coded right now.
            # this should be IMPROVED!

            for ene in energy:

                ene_lo.append(ene - 2.5)
                ene_hi.append(ene + 2.5)

            pol_ang = np.array(f['pol_ang'][()])

            pol_deg = np.array(f['pol_deg'][()])

            bins = np.array(f['bins'][()])

            # get the bin centers as these are where things
            # should be evaluated

            bin_center = 0.5 * (bins[:-1] + bins[1:])

            all_interp = []

            # now we construct a series of interpolation
            # functions that are called during the fit.
            # we use some nice matrix math to handle this

            matrix = np.array(f['matrix'][()])

            print("Computing interpolation grid...")
            pbar = tqdm(total=len(bin_center))
            for i, bm in enumerate(bin_center):

                if not energy_resolved:
                    this_interpolator = FastGridInterpolate((energy, pol_ang, pol_deg),
                                                            matrix[..., i])  # energy, PA, PD

                elif energy_resolved == "pdheavyside":

                    # ene_break = np.arange(100., 501., 5.)

                    rsp_matrix = []
                    for pdlow in range(len(pol_deg)):
                        pdhigh_rsp = []
                        for pdhigh in range(len(pol_deg)):
                            enebreak_mask = energy[:, None][19:-50] < energy  # [19:-50] -> break b/w 100 and 500 keV
                            pdlow_idx = pdlow * np.ones_like(enebreak_mask, dtype=int)
                            pdhigh_idx = pdhigh * np.ones_like(enebreak_mask, dtype=int)
                            pd_idx = np.where(enebreak_mask, pdhigh_idx, pdlow_idx)
                            enebreak_rsp = matrix[np.arange(len(energy)), :, pd_idx, i]
                            pdhigh_rsp.append(enebreak_rsp)
                        rsp_matrix.append(pdhigh_rsp)

                    """rsp_matrix = []  # pdlow, pdhigh, ebreak, energy, PA
                    for pdlow in range(len(pol_deg)):
                        pdhigh_rsp = []
                        for pdhigh in range(len(pol_deg)):
                            enebreak_rsp = []
                            for enebreak in energy[19:-50]:
                                enebreak_rsp.append([matrix[n, :, pdlow, i] if energy[n]<enebreak else matrix[n, :, pdhigh, i] for n in range(len(energy))])
                            pdhigh_rsp.append(enebreak_rsp)
                        rsp_matrix.append(pdhigh_rsp)"""

                    rsp_matrix = np.array(rsp_matrix).transpose(3, 0, 1, 2, 4)  # energy first

                    this_interpolator = FastGridInterpolate((energy, pol_deg, pol_deg, energy[19:-50], pol_ang),
                                                            rsp_matrix)  # energy, pdlow, pdhigh, ebreak, PA

                pbar.update(1)
                all_interp.append(this_interpolator)
            pbar.close()


            # finally we attach all of this to the class

            self._all_interp = all_interp

            self._ene_lo = ene_lo
            self._ene_hi = ene_hi
            self._energy_mid = energy

            self._n_scatter_bins = len(bin_center)
            self._scattering_bins = bin_center
            self._scattering_bins_lo = bins[:-1]
            self._scattering_bins_hi = bins[1:]

    @property
    def ene_lo(self):
        """ 
        The low side of the energy bins

        :returns: 
        :rtype: 

        """
        return self._ene_lo

    @property
    def ene_hi(self):
        """ 
        The high side of the energy bins

        :returns: 
        :rtype: 

        """
        return self._ene_hi

    @property
    def energy_mid(self):
        """ 
        The mid point of the energy bins

        :returns: 
        :rtype: 

        """
        return self._energy_mid

    @property
    def n_scattering_bins(self):
        """
        The number of scattering angle bins

        :returns: 
        :rtype: 

        """

        return self._n_scatter_bins

    @property
    def scattering_bins(self):
        """
        The scattering angle bin CENTERS.

        :returns: 
        :rtype: 

        """

        return self._scattering_bins

    @property
    def scattering_bins_lo(self):
        """
        The low side of the scattering angle bins

        :returns: 
        :rtype: 

        """

        return self._scattering_bins_lo

    @property
    def scattering_bins_hi(self):
        """
        The high side of the scattering angle bins

        :returns: 
        :rtype: 

        """

        return self._scattering_bins_hi

    @property
    def interpolators(self):
        """
        The series of interpolation functions 

        :Returns: 
        :rtype: 

        """

        return self._all_interp
