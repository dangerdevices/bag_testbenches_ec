# -*- coding: utf-8 -*-

import os
import math
from typing import List, Any, Tuple, Dict, Optional

import yaml
import numpy as np
import scipy.constants
import scipy.interpolate

from bag.io import read_yaml, open_file
from bag.core import BagProject, Testbench
from bag.data import Waveform
from bag.data.mos import mos_y_to_ss
from bag.tech.core import SimulationManager
from bag.math.interpolate import LinearInterpolator


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
                vds_max = tb_params['vds_max']
                vds_num = tb_params['vds_num']
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
            elif tb_type == 'tb_noise':
                tb.set_parameter('freq_start', tb_params['freq_start'])
                tb.set_parameter('freq_stop', tb_params['freq_stop'])
                tb.set_parameter('num_per_dec', tb_params['num_per_dec'])
                tb.set_parameter('vbs', tb_params['vbs'])

                vds_min = tb_params['vds_min']
                vds_max = tb_params['vds_max']
                vds_num = tb_params['vds_num']
                vgs_num = tb_params['vgs_num']
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
            vgs_min = wv_max.get_crossing(ibias_min_fg * fg)
            vgs_max = wv_min.get_crossing(ibias_max_fg * fg)
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
        # type: () -> Tuple[List[str], Dict[str, Dict[str, LinearInterpolator]]]
        tb_type = 'tb_sp'
        tb_specs = self.specs[tb_type]
        sch_params = self.specs['sch_params']
        dsn_name_base = self.specs['dsn_name_base']

        fg = sch_params['nf']
        char_freq = tb_specs['tb_params']['sp_freq']

        axis_names = ['corner', 'vds', 'vgs']
        delta_list = [0.1, 1e-6, 1e-6]
        corner_list = None
        total_dict = {}
        for val_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            results = self.get_sim_results(tb_type, val_list)
            ibias = results['ibias']
            if not self.is_nmos(val_list):
                ibias *= -1
            ss_dict = mos_y_to_ss(results, char_freq, fg, ibias)

            if corner_list is None:
                corner_list = results['corner'].tolist()

            points = np.arange(len(corner_list)), results['vds'], results['vgs']

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['ibias']
            order = [swp_vars.index(name) for name in axis_names]
            # just to be safe, we create a list copy to avoid modifying dictionary
            # while iterating over view.
            for key in list(ss_dict.keys()):
                new_data = np.transpose(ss_dict[key], axes=order)
                ss_dict[key] = LinearInterpolator(points, new_data, delta_list, extrapolate=True)

            total_dict[dsn_name] = ss_dict

        return corner_list, total_dict

    def get_integrated_noise(self, fstart, fstop, scale=1.0):
        # type: (Optional[float], Optional[float], float) -> Tuple[List[str], Dict[str, LinearInterpolator]]
        tb_type = 'tb_noise'
        sch_params = self.specs['sch_params']
        dsn_name_base = self.specs['dsn_name_base']

        fg = sch_params['nf']

        axis_names = ['corner', 'vds', 'vgs', 'freq']
        delta_list = [0.1, 1e-6, 1e-6, 0.1]
        corner_list = log_freq = None
        output_dict = {}
        for val_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            results = self.get_sim_results(tb_type, val_list)
            out = np.log(scale / fg * results['idn'] ** 2)

            if corner_list is None:
                corner_list = results['corner'].tolist()
                log_freq = np.log(results['freq'])

            points = np.arange(len(corner_list)), results['vds'], results['vgs'], log_freq

            fstart_log = log_freq[0] if fstart is None else np.log(fstart)
            fstop_log = log_freq[-1] if fstop is None else np.log(fstop)

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['idn']
            order = [swp_vars.index(name) for name in axis_names]
            noise_fun = LinearInterpolator(points, np.transpose(out, axes=order), delta_list, extrapolate=True)
            integ_noise = noise_fun.integrate(fstart_log, fstop_log, axis=-1, logx=True, logy=True)
            output_dict[dsn_name] = integ_noise

        return corner_list, output_dict

    def get_ss_with_noise(self,
                          fstart,  # type: Optional[float]
                          fstop,  # type: Optional[float]
                          scale=1.0,  # type: float
                          temp=300,  # type: float
                          ):
        # type: (...) -> Tuple[List[str], Dict[str, Dict[str, LinearInterpolator]]]
        corner_list, tot_dict = self.get_ss_params()
        _, noise_dict = self.get_integrated_noise(fstart, fstop, scale=scale)

        k = scale * (fstop - fstart) * 4 * scipy.constants.Boltzmann * temp
        for key, val in tot_dict.items():
            gm = val['gm']
            noise_var = noise_dict[key]
            val['gamma_eff'] = noise_var / gm / k

        return corner_list, tot_dict

    def get_dsn_noise_info(self,
                           dsn_name,  # type: str
                           fstart,  # type: Optional[float]
                           fstop,  # type: Optional[float]
                           vgd,  # type: float
                           itarg,  # type: float
                           scale=1.0,  # type: float
                           temp=300,  # type: float
                           ):
        # type: (...) -> Dict[str, Any]
        corner_list, tot_dict = self.get_ss_with_noise(fstart, fstop, scale=scale, temp=temp)
        ss_dict = tot_dict[dsn_name]

        ibias = ss_dict['ibias']
        gamma = ss_dict['gamma_eff']
        gm = ss_dict['gm']

        vgs = ibias.get_input_points(2)
        xmat = np.empty((vgs.size, 3))
        xmat[:, 2] = vgs
        xmat[:, 1] = vgs - vgd

        k = scale * (fstop - fstart) * 4 * scipy.constants.Boltzmann * temp
        info_dict = {'ibias_unit': [], 'gm': [], 'gamma': [], 'vgn': [], 'corners': corner_list}
        for idx, corner in enumerate(corner_list):
            xmat[:, 0] = idx
            ibias_cur = ibias(xmat)
            gm_cur = gm(xmat) * itarg / ibias_cur
            gamma_cur = gamma(xmat)
            vgn = np.sqrt(k * gamma_cur / gm_cur)

            info_dict['ibias_unit'].append(ibias_cur)
            info_dict['gm'].append(gm_cur)
            info_dict['gamma'].append(gamma_cur)
            info_dict['vgn'].append(vgn)

        return info_dict

    def plot_dsn_info(self,
                      dsn_name,  # type: str
                      fstart,  # type: Optional[float]
                      fstop,  # type: Optional[float]
                      vgd,  # type: float
                      itarg,  # type: float
                      scale=1.0,  # type: float
                      temp=300,  # type: float
                      ):
        # type: (...) -> None
        import matplotlib.pyplot as plt

        plt_names = ['vgn', 'gamma', 'gm']
        plt_unit_str = ['nV', '', 'uS']
        plt_unit = [1e-9, 1, 1e-6]
        xname = 'ibias_unit'
        xunit = 1e-6
        xlabel = 'ibias_unit (uA)'

        print(dsn_name)

        info_dict = self.get_dsn_noise_info(dsn_name, fstart, fstop, vgd, itarg, scale=scale, temp=temp)
        corners = info_dict['corners']
        xvec_list = info_dict[xname]

        # figure out X limit
        xmin = xmax = None
        for xvec in xvec_list:
            min_cur = np.min(xvec)
            max_cur = np.max(xvec)
            if xmin is None:
                xmin, xmax = min_cur, max_cur
            else:
                xmin = max(xmin, min_cur)
                xmax = min(xmax, max_cur)

        xfine = np.linspace(xmin, xmax, 100)

        plt.figure(1)
        nplt = len(plt_names)
        ax0 = ax_cur = None
        for idx, (name, unit_str, unit) in enumerate(zip(plt_names, plt_unit_str, plt_unit)):
            if ax0 is None:
                ax0 = ax_cur = plt.subplot(nplt, 1, idx + 1)
            else:
                ax_cur = plt.subplot(nplt, 1, idx + 1, sharex=ax0)
            ax_cur.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2), useMathText=True)
            ax_cur.set_title('%s vs %s' % (name, xname))
            if unit_str:
                ax_cur.set_ylabel('%s (%s)' % (name, unit_str))
            else:
                ax_cur.set_ylabel(name)

            yvec_list = info_dict[name]
            ymin = ymax = None
            for xvec, yvec in zip(xvec_list, yvec_list):
                yfine = scipy.interpolate.interp1d(xvec, yvec)(xfine)
                ymin_cur = np.min(yfine)
                ymax_cur = np.max(yfine)
                if ymin is None:
                    ymin, ymax = ymin_cur, ymax_cur
                else:
                    ymin = min(ymin, ymin_cur)
                    ymax = max(ymax, ymax_cur)

            for xvec, yvec, corner in zip(xvec_list, yvec_list, corners):
                ax_cur.plot(xvec / xunit, yvec / unit, 'o-', label=corner)

            ax_cur.set_xlim([xmin / xunit, xmax / xunit])
            ax_cur.set_ylim([ymin / unit, ymax / unit])
            ax_cur.legend()

        if ax_cur is not None:
            ax_cur.set_xlabel(xlabel)

        plt.show()


if __name__ == '__main__':

    config_file = 'mos_char_specs/mos_char_nch_stack.yaml'

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    sim = MOSCharSim(bprj, config_file)

    # sim.run_lvs_rcx(tb_type='tb_ibias')
    # sim.run_simulations('tb_ibias')
    # sim.process_ibias_data()

    # sim.run_simulations('tb_sp')
    # sim.run_simulations('tb_noise')

    fc = 100e3
    fbw = 500
    dname = 'MOS_NCH_STACK_intent_svt_l_90n'
    sim.plot_dsn_info(dname, fc - fbw / 2, fc + fbw / 2, 0.0, 1e-6)
