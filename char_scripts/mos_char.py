# -*- coding: utf-8 -*-

import os
import math
from typing import List, Any, Tuple, Dict

import yaml
import numpy as np

from bag.io import read_yaml, open_file
from bag.core import BagProject, Testbench
from bag.data import Waveform
from bag.data.mos import mos_y_to_ss
from bag.tech.core import SimulationManager


class MOSCharSim(SimulationManager):
    """A class that handles transistor characterization."""
    def __init__(self, prj, spec_file):
        super(MOSCharSim, self).__init__(prj, spec_file)

    def get_sch_lay_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Tuple[Dict[str, Any], Dict[str, Any]]
        sch_params = self.specs['sch_params'].copy()
        lay_params = self.specs['layout_params'].copy()
        for var, val in zip(self.swp_var_list, val_list):
            sch_params[var] = val

        lay_params['mos_type'] = sch_params['mos_type']
        lay_params['lch'] = sch_params['l']
        lay_params['w'] = sch_params['w']
        lay_params['threshold'] = sch_params['intent']
        lay_params['stack'] = sch_params['stack']
        lay_params['fg'] = sch_params['nf']
        lay_params['fg_dum'] = sch_params['ndum']
        return sch_params, lay_params

    def is_nmos(self, val_list):
        # type: (Tuple[Any, ...]) -> bool
        """Given current schematic parameter values, returns True if we're working with NMOS.."""
        try:
            # see if mos_type is one of the sweep.
            idx = self.swp_var_list.index('mos_type')
            return val_list[idx] == 'nch'
        except:
            # mos_type is not one of the sweep.
            return self.specs['sch_params']['mos_type'] == 'nch'

    def get_vgs_specs(self):
        # type: () -> Dict[str, Any]
        """laods VGS specifications from file and return it as dictionary."""
        vgs_file = os.path.join(self.specs['root_dir'], self.specs['vgs_file'])
        return read_yaml(vgs_file)

    def configure_tb(self, tb_type, tb, val_list):
        # type: (str, Testbench, Tuple[Any, ...]) -> None

        tb_specs = self.specs[tb_type]
        sim_envs = self.specs['sim_envs']
        view_name = self.specs['view_name']
        impl_lib = self.specs['impl_lib']
        dsn_name_base = self.specs['dsn_name_base']

        tb_params = tb_specs['tb_params']
        dsn_name = self.get_instance_name(dsn_name_base, val_list)
        is_nmos = self.is_nmos(val_list)

        tb.set_simulation_environments(sim_envs)
        tb.set_simulation_view(impl_lib, dsn_name, view_name)

        if tb_type == 'tb_ibias':
            tb.set_parameter('vgs_num', tb_params['vgs_num'])

            # handle VGS sign for nmos/pmos
            vgs_max = tb_params['vgs_max']
            if is_nmos:
                tb.set_parameter('vs', 0.0)
                tb.set_parameter('vgs_start', 0.0)
                tb.set_parameter('vgs_stop', vgs_max)
            else:
                tb.set_parameter('vs', vgs_max)
                tb.set_parameter('vgs_start', -vgs_max)
                tb.set_parameter('vgs_stop', 0.0)
        else:
            vgs_info = self.get_vgs_specs()
            vgs_start, vgs_stop = vgs_info[dsn_name]

            if tb_type == 'tb_sp':
                tb.set_parameter('vgs_num', tb_params['vgs_num'])
                tb.set_parameter('sp_freq', tb_params['sp_freq'])
                tb.set_parameter('vbs', tb_params['vbs'])

                vds_min = tb_params['vds_min']
                vds_num = tb_params['vds_num']
                tb.set_parameter('vgs_start', vgs_start)
                tb.set_parameter('vgs_stop', vgs_stop)
                # handle VDS/VGS sign for nmos/pmos
                if is_nmos:
                    vds_vals = np.linspace(vds_min, vgs_stop, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vb_dc', 0)
                else:
                    vds_vals = np.linspace(vgs_start, -vds_min, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vb_dc', abs(vgs_start))
            elif tb_type == 'tb_noise':
                tb.set_parameter('freq_start', tb_params['freq_start'])
                tb.set_parameter('freq_stop', tb_params['freq_stop'])
                tb.set_parameter('num_per_dec', tb_params['num_per_dec'])
                tb.set_parameter('vbs', tb_params['vbs'])

                vds_min = tb_params['vds_min']
                vds_num = tb_params['vds_num']
                vgs_num = tb_params['vgs_num']
                vgs_vals = np.linspace(vgs_start, vgs_stop, vgs_num + 1)
                # handle VDS/VGS sign for nmos/pmos
                if is_nmos:
                    vds_vals = np.linspace(vds_min, vgs_stop, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vgs', values=vgs_vals)
                    tb.set_sweep_parameter('vb_dc', 0)
                else:
                    vds_vals = np.linspace(vgs_start, -vds_min, vds_num + 1)
                    tb.set_sweep_parameter('vds', values=vds_vals)
                    tb.set_sweep_parameter('vgs', values=vgs_vals)
                    tb.set_sweep_parameter('vb_dc', abs(vgs_start))
            else:
                raise ValueError('Unknown testbench type: %s' % tb_type)

    def process_ibias_data(self, write=True):
        # type: () -> None
        tb_type = 'tb_ibias'
        tb_specs = self.specs[tb_type]
        dsn_name_base = self.specs['dsn_name_base']
        root_dir = self.specs['root_dir']
        vgs_file = self.specs['vgs_file']
        sch_params = self.specs['sch_params']

        fg = sch_params['nf']
        ibias_min_fg = tb_specs['ibias_min_fg']
        ibias_max_fg = tb_specs['ibias_max_fg']
        vgs_res = tb_specs['vgs_resolution']

        ans = {}
        for val_list in self.get_combinations_iter():
            # invert PMOS ibias sign
            ibias_sgn = 1.0 if self.is_nmos(val_list) else -1.0
            results = self.get_sim_results(tb_type, val_list)

            # assume first sweep parameter is corner, second sweep parameter is vgs
            corner_idx = results['sweep_params']['ibias'].index('corner')
            vgs = results['vgs']
            ibias = results['ibias'] * ibias_sgn  # type: np.ndarray

            wv_max = Waveform(vgs, np.amax(ibias, corner_idx), 1e-6, order=2)
            wv_min = Waveform(vgs, np.amin(ibias, corner_idx), 1e-6, order=2)
            vgs_min = wv_min.get_crossing(ibias_min_fg * fg)
            vgs_max = wv_max.get_crossing(ibias_max_fg * fg)
            if vgs_min > vgs_max:
                vgs_min, vgs_max = vgs_max, vgs_min
            vgs_min = math.floor(vgs_min / vgs_res) * vgs_res
            vgs_max = math.ceil(vgs_max / vgs_res) * vgs_res

            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            print('%s: vgs = [%.4g, %.4g]' % (dsn_name, vgs_min, vgs_max))
            ans[dsn_name] = [vgs_min, vgs_max]

        if write:
            vgs_file = os.path.join(root_dir, vgs_file)
            with open_file(vgs_file, 'w') as f:
                yaml.dump(ans, f)

    def get_ss_params(self):
        # type: () -> Tuple[Tuple[np.array, ...], List[Dict[str, np.ndarray]]]
        tb_type = 'tb_sp'
        tb_specs = self.specs[tb_type]
        sch_params = self.specs['sch_params']

        fg = sch_params['nf']
        char_freq = tb_specs['tb_params']['sp_freq']

        axis_names = ['corner', 'vds', 'vgs']
        xvals = None
        ss_list = []
        for val_list in self.get_combinations_iter():
            results = self.get_sim_results(tb_type, val_list)
            ibias = results['ibias']
            ss_dict = mos_y_to_ss(results, char_freq, fg, ibias)

            if xvals is None:
                xvals = results['corner'], results['vds'], results['vgs']

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['ibias']
            order = [swp_vars.index(name) for name in axis_names]
            # just to be safe, we create a list copy to avoid modifying dictionary
            # while iterating over view.
            for key in list(ss_dict.keys()):
                ss_dict[key] = np.transpose(ss_dict[key], axes=order)

            ss_list.append(ss_dict)

        return xvals, ss_list

    def get_noise_psd(self):
        # type: () -> Tuple[Tuple[np.array, ...], List[Dict[str, np.ndarray]]]
        tb_type = 'tb_noise'
        sch_params = self.specs['sch_params']

        fg = sch_params['nf']

        axis_names = ['corner', 'vds', 'vgs', 'freq']
        output_list = []
        xvals = None
        for val_list in self.get_combinations_iter():
            results = self.get_sim_results(tb_type, val_list)
            out = results['idn']**2 / fg

            if xvals is None:
                xvals = results['corner'], results['vds'], results['vgs'], results['freq']

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['idn']
            order = [swp_vars.index(name) for name in axis_names]
            # just to be safe, we create a list copy to avoid modifying dictionary
            # while iterating over view.
            out = np.transpose(out, axes=order)
            output_list.append(out)

        return xvals, output_list


if __name__ == '__main__':

    config_file = 'mos_char_specs/mos_char_nch.yaml'

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    sim = MOSCharSim(bprj, config_file)
    # sim.process_ibias_data()
