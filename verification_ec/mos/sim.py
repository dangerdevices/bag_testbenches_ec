# -*- coding: utf-8 -*-

"""This package contains measurement class for transistors."""

from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, Sequence

import numpy as np

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
    """

    def __init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list):
        # type: (str, str, str, Dict[str, Any], Dict[str, str], Sequence[Tuple[str, str]]) -> None
        MeasurementManager.__init__(self, data_dir, meas_name, impl_lib, specs, wrapper_lookup, sim_view_list)

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
            pass

        return False, '', {}

    def process_ibias_data(self, tb_specs):
        layout_params = self.specs['layout_params']

        fg = layout_params['fg']
        ibias_min_fg = tb_specs['ibias_min_fg']
        ibias_max_fg = tb_specs['ibias_max_fg']
        vgs_res = tb_specs['vgs_resolution']

        ans = {}
        for val_list in self.get_combinations_iter():
            # invert PMOS ibias sign
            is_nmos = self.is_nmos(val_list)
            ibias_sgn = 1.0 if is_nmos else -1.0
            results = self.get_sim_results(tb_type, val_list)

            # assume first sweep parameter is corner, second sweep parameter is vgs
            corner_idx = results['sweep_params']['ibias'].index('corner')
            vgs = results['vgs']
            ibias = results['ibias'] * ibias_sgn  # type: np.ndarray

            wv_max = Waveform(vgs, np.amax(ibias, corner_idx), 1e-6, order=2)
            wv_min = Waveform(vgs, np.amin(ibias, corner_idx), 1e-6, order=2)
            vgs1 = wv_max.get_crossing(ibias_min_fg * fg)
            if vgs1 is None:
                vgs1 = vgs[0] if is_nmos else vgs[-1]
            vgs2 = wv_min.get_crossing(ibias_max_fg * fg)
            if vgs2 is None:
                vgs2 = vgs[-1] if is_nmos else vgs[0]

            if is_nmos:
                vgs_min, vgs_max = vgs1, vgs2
            else:
                vgs_min, vgs_max = vgs2, vgs1

            vgs_min = math.floor(vgs_min / vgs_res) * vgs_res
            vgs_max = math.ceil(vgs_max / vgs_res) * vgs_res

            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            print('%s: vgs = [%.4g, %.4g]' % (dsn_name, vgs_min, vgs_max))
            ans[dsn_name] = [vgs_min, vgs_max]

        if write:
            vgs_file = os.path.join(root_dir, vgs_file)
            with open_file(vgs_file, 'w') as f:
                yaml.dump(ans, f)