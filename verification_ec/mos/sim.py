# -*- coding: utf-8 -*-

"""This package contains measurement class for transistors."""

from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, Sequence

import os
import math

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bag.io.sim_data import save_sim_results
from bag.simulation.core import MeasurementManager

if TYPE_CHECKING:
    from bag.core import Testbench


class MosCharSS(MeasurementManager):
    """This class measures small signal parameters of a transistor using Y parameter fitting.

    This measurement is perform as follows:

    1. First, given a user specified current density range, we perform a DC current measurement
       to find the range of vgs needed across corners to cover that range.
    2. Then, we run a S parameter simulation and record Y parameter values at various bias points.
    3. If user specify a noise testbench, a noise simulation will be run at the same bias points
       as S parameter simulation to characterize transistor noise.

    Parameters
    ----------
    data_dir : str
        Simulation data directory.
    meas_name : str
        measurement setup name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        the measurement specification dictionary.
    wrapper_lookup : Dict[str, str]
        the DUT wrapper cell name lookup table.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    """

    def __init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list):
        # type: (str, str, str, Dict[str, Any], Dict[str, str], Sequence[Tuple[str, str]], Sequence[str]) -> None
        MeasurementManager.__init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list, env_list)

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'ibias'

    def get_testbench_info(self, state, prev_output):
        # type: (str, Optional[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]

        tb_type = state
        tb_specs = self.specs['testbenches'][tb_type]
        wrapper_type = tb_specs['wrapper_type']
        tb_params = self.get_default_tb_sch_params(tb_type, wrapper_type)
        tb_name = self.get_testbench_name(tb_type)

        return tb_name, tb_type, tb_specs, tb_params

    def setup_testbench(self, state, tb, tb_specs):
        # type: (str, Testbench, Dict[str, Any]) -> None

        is_nmos = self.specs['is_nmos']

        vgs_num = tb_specs['vgs_num']

        if state == 'ibias':
            vgs_max = tb_specs['vgs_max']

            tb.set_parameter('vgs_num', vgs_num)

            # handle VGS sign for nmos/pmos
            if is_nmos:
                tb.set_parameter('vs', 0.0)
                tb.set_parameter('vgs_start', 0.0)
                tb.set_parameter('vgs_stop', vgs_max)
            else:
                tb.set_parameter('vs', vgs_max)
                tb.set_parameter('vgs_start', -vgs_max)
                tb.set_parameter('vgs_stop', 0.0)
        else:
            vbs_val = tb_specs['vbs']
            vds_min = tb_specs['vds_min']
            vds_max = tb_specs['vds_max']
            vds_num = tb_specs['vds_num']

            vgs_start, vgs_stop = self.get_state_output('ibias')['vgs_range']

            # handle VBS sign and set parameters.
            if isinstance(vbs_val, list):
                if is_nmos:
                    vbs_val = sorted((-abs(v) for v in vbs_val))
                else:
                    vbs_val = sorted((abs(v) for v in vbs_val))
                tb.set_sweep_parameter('vbs', values=vbs_val)
            else:
                if is_nmos:
                    vbs_val = -abs(vbs_val)
                else:
                    vbs_val = abs(vbs_val)
                tb.set_parameter('vbs', vbs_val)

            if state == 'sp':
                sp_freq = tb_specs['sp_freq']

                tb.set_parameter('vgs_num', vgs_num)
                tb.set_parameter('sp_freq', sp_freq)

                tb.set_parameter('vgs_start', vgs_start)
                tb.set_parameter('vgs_stop', vgs_stop)
                # handle VDS/VGS sign for nmos/pmos
                if is_nmos:
                    vds_vals = np.linspace(vds_min, vds_max, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_parameter('vb_dc', 0)
                else:
                    vds_vals = np.linspace(-vds_max, -vds_min, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_parameter('vb_dc', abs(vgs_start))
            elif state == 'noise':
                freq_start = tb_specs['freq_start']
                freq_stop = tb_specs['freq_stop']
                num_per_dec = tb_specs['num_per_dec']

                tb.set_parameter('freq_start', freq_start)
                tb.set_parameter('freq_stop', freq_stop)
                tb.set_parameter('num_per_dec', num_per_dec)

                vgs_vals = np.linspace(vgs_start, vgs_stop, vgs_num + 1)
                # handle VDS/VGS sign for nmos/pmos
                if is_nmos:
                    vds_vals = np.linspace(vds_min, vds_max, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vgs', values=vgs_vals)
                    tb.set_parameter('vb_dc', 0)
                else:
                    vds_vals = np.linspace(-vds_max, -vds_min, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vgs', values=vgs_vals)
                    tb.set_parameter('vb_dc', abs(vgs_start))

    def process_output(self, state, data, tb_specs):
        # type: (str, Dict[str, Any], Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]

        if state == 'ibias':
            done = False
            next_state = 'sp'
            output = self.process_ibias_data(data, tb_specs)
        elif state == 'sp':
            testbenches = self.specs['testbenches']
            if 'noise' in testbenches:
                done = False
                next_state = 'noise'
            else:
                done = True
                next_state = ''
            output = dict(ss_file=self.process_sp_data(data, tb_specs))
        elif state == 'noise':
            done = True
            next_state = ''
            output = dict(noise_file='')
        else:
            raise ValueError('Unknown state: %s' % state)

        return done, next_state, output

    def process_sp_data(self, data, tb_specs):
        fg = self.specs['fg']
        is_nmos = self.specs['is_nmos']

        cfit_method = tb_specs['cfit_method']
        sp_freq = tb_specs['sp_freq']

        axis_names = ['corner', 'vbs', 'vds', 'vgs']

        ibias = data['ibias']
        if not is_nmos:
            ibias = -1 * ibias
        ss_dict = self.mos_y_to_ss(data, sp_freq, fg, ibias, cfit_method=cfit_method)

        ss_swp_names = [name for name in axis_names[1:] if name in data]
        swp_corner = ('corner' in data)
        if swp_corner:
            corner_list = data['corner']
        else:
            corner_list = [self.env_list[0]]

        # rearrange array axis
        swp_vars = data['sweep_params']['ibias']
        order = [swp_vars.index(name) for name in axis_names if name in swp_vars]

        # construct new SS parameter result dictionary
        new_swp_info = {}
        new_swp_vars = ['corner', ] + ss_swp_names
        new_result = {'sweep_params': new_swp_info, 'corner': corner_list}
        for name in ss_swp_names:
            new_result[name] = data[name]
        for key, val in ss_dict.items():
            new_data = np.transpose(val, axes=order)
            if not swp_corner:
                new_data = new_data[np.newaxis, ...]
            new_swp_info[key] = new_swp_vars
            new_result[key] = new_data

        # save SS parameters
        file_name = os.path.join(self.data_dir, 'ss_params.hdf5')
        save_sim_results(new_result, file_name)
        return file_name

    def process_ibias_data(self, data, tb_specs):
        fg = self.specs['fg']
        is_nmos = self.specs['is_nmos']

        ibias_min_fg = tb_specs['ibias_min_fg']
        ibias_max_fg = tb_specs['ibias_max_fg']
        vgs_res = tb_specs['vgs_resolution']

        # invert PMOS ibias sign
        ibias_sgn = 1.0 if is_nmos else -1.0

        vgs = data['vgs']
        ibias = data['ibias'] * ibias_sgn  # type: np.ndarray

        # assume first sweep parameter is corner, second sweep parameter is vgs
        try:
            corner_idx = data['sweep_params']['ibias'].index('corner')
            ivec_max = np.amax(ibias, corner_idx)
            ivec_min = np.amin(ibias, corner_idx)
        except ValueError:
            ivec_max = ivec_min = ibias

        vgs1 = self._get_best_crossing(vgs, ivec_max, ibias_min_fg * fg)
        vgs2 = self._get_best_crossing(vgs, ivec_min, ibias_max_fg * fg)

        vgs_min = min(vgs1, vgs2)
        vgs_max = max(vgs1, vgs2)

        vgs_min = math.floor(vgs_min / vgs_res) * vgs_res
        vgs_max = math.ceil(vgs_max / vgs_res) * vgs_res

        return dict(vgs_range=[vgs_min, vgs_max])

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop

    @classmethod
    def mos_y_to_ss(cls, sim_data, char_freq, fg, ibias, cfit_method='average'):
        # type: (Dict[str, np.ndarray], float, int, np.ndarray, str) -> Dict[str, np.ndarray]
        """Convert transistor Y parameters to small-signal parameters.

        This function computes MOSFET small signal parameters from 3-port
        Y parameter measurements done on gate, drain and source, with body
        bias fixed.  This functions fits the Y parameter to a capcitor-only
        small signal model using least-mean-square error.

        Parameters
        ----------
        sim_data : Dict[str, np.ndarray]
            A dictionary of Y parameters values stored as complex numpy arrays.
        char_freq : float
            the frequency Y parameters are measured at.
        fg : int
            number of transistor fingers used for the Y parameter measurement.
        ibias : np.ndarray
            the DC bias current of the transistor.  Always positive.
        cfit_method : str
            method used to extract capacitance from Y parameters.  Currently
            supports 'average' or 'worst'

        Returns
        -------
        ss_dict : Dict[str, np.ndarray]
            A dictionary of small signal parameter values stored as numpy
            arrays.  These values are normalized to 1-finger transistor.
        """
        w = 2 * np.pi * char_freq

        gm = (sim_data['y21'].real - sim_data['y31'].real) / 2.0
        gds = (sim_data['y22'].real - sim_data['y32'].real) / 2.0
        gb = (sim_data['y33'].real - sim_data['y23'].real) / 2.0 - gm - gds

        cgd12 = -sim_data['y12'].imag / w
        cgd21 = -sim_data['y21'].imag / w
        cgs13 = -sim_data['y13'].imag / w
        cgs31 = -sim_data['y31'].imag / w
        cds23 = -sim_data['y23'].imag / w
        cds32 = -sim_data['y32'].imag / w
        cgg = sim_data['y11'].imag / w
        cdd = sim_data['y22'].imag / w
        css = sim_data['y33'].imag / w

        if cfit_method == 'average':
            cgd = (cgd12 + cgd21) / 2
            cgs = (cgs13 + cgs31) / 2
            cds = (cds23 + cds32) / 2
        elif cfit_method == 'worst':
            cgd = np.maximum(cgd12, cgd21)
            cgs = np.maximum(cgs13, cgs31)
            cds = np.maximum(cds23, cds32)
        else:
            raise ValueError('Unknown cfit_method = %s' % cfit_method)

        cgb = cgg - cgd - cgs
        cdb = cdd - cds - cgd
        csb = css - cgs - cds

        ibias = ibias / fg  # type: np.ndarray
        gm = gm / fg  # type: np.ndarray
        gds = gds / fg  # type: np.ndarray
        gb = gb / fg  # type: np.ndarray
        cgd = cgd / fg  # type: np.ndarray
        cgs = cgs / fg  # type: np.ndarray
        cds = cds / fg  # type: np.ndarray
        cgb = cgb / fg  # type: np.ndarray
        cdb = cdb / fg  # type: np.ndarray
        csb = csb / fg  # type: np.ndarray

        return dict(
            ibias=ibias,
            gm=gm,
            gds=gds,
            gb=gb,
            cgd=cgd,
            cgs=cgs,
            cds=cds,
            cgb=cgb,
            cdb=cdb,
            csb=csb,
        )
