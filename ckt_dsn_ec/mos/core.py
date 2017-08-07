# -*- coding: utf-8 -*-

"""This module contains essential classes/methods for transistor characterization."""

import os
import math
from typing import List, Any, Tuple, Dict, Optional, Union, Sequence

import yaml
import numpy as np
import scipy.constants
import scipy.interpolate

from bag.io import read_yaml, open_file
from bag.core import Testbench, BagProject
from bag.data import Waveform
from bag.data.mos import mos_y_to_ss
from bag.tech.core import SimulationManager
from bag.math.dfun import VectorDiffFunction, DiffFunction
from bag.math.interpolate import LinearInterpolator


class MOSCharSS(SimulationManager):
    """A class that handles transistor small-signal parameters characterization.

    This class characterizes transistor given a range of current-per-finger
    specifications.  It uses AnalogBase to draw the transistor layout, and
    currently it sweeps vgs/vds (not vbs yet) and also characterizes noise
    for a given range of frequency.

    In addition to entries required by SimulationManager, The YAML specification
    file must have the following entries:

    root_dir :
        directory to save simulation files.
    vgs_file :
        file to save vgs sweep information.
        Given current-per-finger spec, this class will figure out
        the proper vgs range to sweep.  This range is saved to this file.
    sch_params :
        Dictionary of default schematic parameters.
    layout_params :
        Dictionary of default layout parameters.
    dsn_name_base :
        the generated transistor cellview base name.
    sim_envs :
        List of simulation environment names.
    view_name :
        Extracted cell view name.
    impl_lib :
        library to put all generated cellviews.
    tb_ibias :
        bias current testbench parameters.  This testbench is used to
        find the proper range of vgs to characterize the transistor with.
        It should have the following entries:

        vgs_max :
            magnitude of maximum vgs value.
        vgs_num :
            number of vgs points.
    """

    def __init__(self, prj, spec_file):
        # type: (Optional[BagProject], str) -> None
        super(MOSCharSS, self).__init__(prj, spec_file)

    @classmethod
    def get_ss_sweep_names(cls):
        # type: () -> List[str]
        return ['vds', 'vgs']

    @classmethod
    def get_ss_output_names(cls):
        # type: () -> List[str]
        return ['ibias', 'gm', 'gds', 'gb', 'cgd', 'cgs', 'cds', 'cgb', 'cdb', 'csb', 'gamma']

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

    def get_default_dsn_value(self, name):
        # type: (str) -> Any
        """Returns default design parameter value."""
        return self.specs['sch_params'][name]

    def is_nmos(self, val_list):
        # type: (Tuple[Any, ...]) -> bool
        """Given current schematic parameter values, returns True if we're working with NMOS.."""
        try:
            # see if mos_type is one of the sweep.
            idx = self.swp_var_list.index('mos_type')
            return val_list[idx] == 'nch'
        except ValueError:
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

    def _get_ss_params(self):
        # type: () -> Tuple[List[str], Dict[str, Dict[str, List[LinearInterpolator]]]]
        tb_type = 'tb_sp'
        tb_specs = self.specs[tb_type]
        sch_params = self.specs['sch_params']
        dsn_name_base = self.specs['dsn_name_base']

        fg = sch_params['nf']
        char_freq = tb_specs['tb_params']['sp_freq']

        axis_names = ['corner', 'vds', 'vgs']
        delta_list = [1e-6, 1e-6]
        corner_list = None
        corner_sort_arg = None  # type: Sequence[int]
        total_dict = {}
        for val_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            results = self.get_sim_results(tb_type, val_list)
            ibias = results['ibias']
            if not self.is_nmos(val_list):
                ibias *= -1
            ss_dict = mos_y_to_ss(results, char_freq, fg, ibias)

            if corner_list is None:
                corner_list = results['corner']
                corner_sort_arg = np.argsort(corner_list)  # type: Sequence[int]
                corner_list = corner_list[corner_sort_arg].tolist()

            points = results['vds'], results['vgs']

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['ibias']
            order = [swp_vars.index(name) for name in axis_names]
            # just to be safe, we create a list copy to avoid modifying dictionary
            # while iterating over view.
            for key in list(ss_dict.keys()):
                new_data = np.transpose(ss_dict[key], axes=order)
                fun_list = []
                for idx in corner_sort_arg:
                    fun_list.append(LinearInterpolator(points, new_data[idx, ...], delta_list, extrapolate=True))
                ss_dict[key] = fun_list

            # derived ss parameters
            self._add_derived_ss_params(ss_dict)

            total_dict[dsn_name] = ss_dict

        return corner_list, total_dict

    @classmethod
    def _add_derived_ss_params(cls, ss_dict):
        cgdl = ss_dict['cgd']
        cgsl = ss_dict['cgs']
        cgbl = ss_dict['cgb']
        cdsl = ss_dict['cds']
        cdbl = ss_dict['cdb']
        csbl = ss_dict['csb']

        ss_dict['cgg'] = [cgd + cgs + cgb for (cgd, cgs, cgb) in zip(cgdl, cgsl, cgbl)]
        ss_dict['cdd'] = [cgd + cds + cdb for (cgd, cds, cdb) in zip(cgdl, cdsl, cdbl)]
        ss_dict['css'] = [cds + cds + csb for (cds, cds, csb) in zip(cgsl, cdsl, csbl)]

    def _get_integrated_noise(self, fstart, fstop, scale=1.0):
        # type: (Optional[float], Optional[float], float) -> Tuple[List[str], Dict[str, List[LinearInterpolator]]]
        tb_type = 'tb_noise'
        sch_params = self.specs['sch_params']
        dsn_name_base = self.specs['dsn_name_base']

        fg = sch_params['nf']

        axis_names = ['corner', 'vds', 'vgs', 'freq']
        delta_list = [1e-6, 1e-6, 1e-3]
        corner_list = log_freq = None
        corner_sort_arg = None  # type: Sequence[int]
        output_dict = {}
        for val_list in self.get_combinations_iter():
            dsn_name = self.get_instance_name(dsn_name_base, val_list)
            results = self.get_sim_results(tb_type, val_list)
            out = np.log(scale / fg * results['idn'] ** 2)

            if corner_list is None:
                corner_list = results['corner']
                corner_sort_arg = np.argsort(corner_list)  # type: Sequence[int]
                corner_list = corner_list[corner_sort_arg].tolist()
                log_freq = np.log(results['freq'])

            points = results['vds'], results['vgs'], log_freq

            fstart_log = log_freq[0] if fstart is None else np.log(fstart)
            fstop_log = log_freq[-1] if fstop is None else np.log(fstop)

            # rearrange array axis
            sweep_params = results['sweep_params']
            swp_vars = sweep_params['idn']
            order = [swp_vars.index(name) for name in axis_names]
            fun_list = []
            out_trans = np.transpose(out, axes=order)
            for idx in corner_sort_arg:
                noise_fun = LinearInterpolator(points, out_trans[idx, ...], delta_list, extrapolate=True)
                integ_noise = noise_fun.integrate(fstart_log, fstop_log, axis=-1, logx=True, logy=True)
                fun_list.append(integ_noise)
            output_dict[dsn_name] = fun_list

        return corner_list, output_dict

    def get_ss_info(self,
                    fstart,  # type: Optional[float]
                    fstop,  # type: Optional[float]
                    scale=1.0,  # type: float
                    temp=300,  # type: float
                    ):
        # type: (...) -> Tuple[List[str], Dict[str, Dict[str, List[LinearInterpolator]]]]
        corner_list, tot_dict = self._get_ss_params()
        _, noise_dict = self._get_integrated_noise(fstart, fstop, scale=scale)

        k = scale * (fstop - fstart) * 4 * scipy.constants.Boltzmann * temp
        for key, val in tot_dict.items():
            gm = val['gm']
            noise_var = noise_dict[key]
            val['gamma'] = [nf / gmf / k for nf, gmf in zip(noise_var, gm)]

        return corner_list, tot_dict


class MOSDBDiscrete(object):
    """Transistor small signal parameters database with discrete width choices.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    width_list : List[Union[float, int]]
        list of valid widths.
    spec_list : List[str]
        list of specification file locations corresponding to widths.
    width_res : Union[float, int]
        width resolution.
    noise_fstart : float
        noise integration frequency lower bound.
    noise_fstop : float
        noise integration frequency upper bound.
    noise_scale : float
        noise integration scaling factor.
    noise_temp : float
        noise temperature.
    """

    def __init__(self,
                 width_list,  # type: List[Union[float, int]]
                 spec_list,  # type: List[str]
                 width_res,  # type: Union[float, int]
                 noise_fstart,  # type: float
                 noise_fstop,  # type: float
                 noise_scale=1.0,  # type: float
                 noise_temp=300,  # type: float
                 ):
        # type: (...) -> None
        # error checking
        if len(width_list) != len(spec_list):
            raise ValueError('width_list and spec_list length mismatch.')
        if not width_list:
            raise ValueError('Must have at least one entry.')

        self._width_res = width_res
        self._width_list = [int(round(w / width_res)) for w in width_list]

        self._sim_envs = None
        self._sim_list = []
        self._ss_list = []
        for spec in spec_list:
            sim = MOSCharSS(None, spec)
            corners, ss_dict = sim.get_ss_info(noise_fstart, noise_fstop, scale=noise_scale, temp=noise_temp)
            if self._sim_envs is None:
                self._sim_envs = corners
            elif self._sim_envs != corners:
                raise ValueError('Simulation environments mismatch between given specs.')

            self._sim_list.append(sim)
            self._ss_list.append(ss_dict)

        self._env_list = self._sim_envs
        self._cur_idx = 0
        self._dsn_params = dict(w=width_list[0])
        self._swp_names = self._sim_list[0].get_ss_sweep_names()
        self._fun_names = self._sim_list[0].get_ss_output_names()

    @property
    def env_list(self):
        # type: () -> List[str]
        """The list of simulation environments to consider."""
        return self._env_list

    @env_list.setter
    def env_list(self, new_env_list):
        # type: (List[str]) -> None
        """Sets the list of simulation environments to consider."""
        self._env_list = new_env_list

    @property
    def dsn_params(self):
        # type: () -> Tuple[str, ...]
        """List of design parameters."""
        return self._sim_list[self._cur_idx].swp_var_list

    def get_default_dsn_value(self, var):
        # type: (str) -> Any
        """Returns the default design parameter values."""
        return self._sim_list[self._cur_idx].get_default_dsn_value(var)

    def get_dsn_param_values(self, var):
        # type: (str) -> List[Any]
        """Returns a list of valid design parameter values."""
        return self._sim_list[self._cur_idx].get_swp_var_values(var)

    def set_dsn_params(self, **kwargs):
        # type: (**kwargs) -> None
        """Set the design parameters for which this database will query for."""
        self._dsn_params.update(kwargs)
        w_unit = int(round(kwargs['w'] / self._width_res))
        self._cur_idx = self._width_list.index(w_unit)

    def _get_dsn_name(self, **kwargs):
        # type: (**kwargs) -> str
        if kwargs:
            self.set_dsn_params(**kwargs)
        dsn_name = self._sim_list[self._cur_idx].get_design_name(self._dsn_params)

        if dsn_name not in self._ss_list[self._cur_idx]:
            raise ValueError('Unknown design name: %s.  Did you set design parameters?' % dsn_name)

        return dsn_name

    def get_function_list(self, name, **kwargs):
        # type: (str, **kwargs) -> List[DiffFunction]
        """Returns a list of functions, one for each simulation environment, for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        **kwargs :
            design parameter values.

        Returns
        -------
        output : Union[RegGridInterpVectorFunction, RegGridInterpFunction]
            the output vector function.
        """
        dsn_name = self._get_dsn_name(**kwargs)
        cur_dict = self._ss_list[self._cur_idx][dsn_name]
        fun_list = []
        for env in self.env_list:
            try:
                env_idx = self._sim_envs.index(env)
            except ValueError:
                raise ValueError('environment %s not found.' % env)

            fun_list.append(cur_dict[name][env_idx])
        return fun_list

    def get_function(self, name, env='', **kwargs):
        # type: (str, str, **kwargs) -> Union[VectorDiffFunction, DiffFunction]
        """Returns a function for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        env : str
            if not empty, we will return function for just the given simulation environment.
        **kwargs :
            design parameter values.

        Returns
        -------
        output : Union[RegGridInterpVectorFunction, RegGridInterpFunction]
            the output vector function.
        """
        if not env:
            return VectorDiffFunction(self.get_function_list(name, **kwargs))
        else:
            dsn_name = self._get_dsn_name(**kwargs)
            cur_dict = self._ss_list[self._cur_idx][dsn_name]
            try:
                env_idx = self._sim_envs.index(env)
            except ValueError:
                raise ValueError('environment %s not found.' % env)

            return cur_dict[name][env_idx]

    def get_fun_sweep_params(self, **kwargs):
        # type: (**kwargs) -> Tuple[List[str], List[Tuple[float, float]]]
        """Returns interpolation function sweep parameter names and values.

        Parameters
        ----------
        **kwargs :
            design parameter values.

        Returns
        -------
        sweep_params : List[str]
            list of parameter names.
        sweep_range : List[Tuple[float, float]]
            list of parameter range
        """
        dsn_name = self._get_dsn_name(**kwargs)
        sample_fun = self._ss_list[self._cur_idx][dsn_name]['gm'][0]

        return self._swp_names, sample_fun.input_ranges

    def get_fun_arg(self, **kwargs):
        # type: (**kwargs) -> np.multiarray.ndarray
        """Convert keyword arguments to function argument."""
        return np.array([kwargs[key] for key in self._swp_names])

    def get_fun_arg_index(self, name):
        # type: (str) -> int
        """Returns the function input argument index for the given variable"""
        return self._swp_names.index(name)

    def query(self, **kwargs):
        # type: (**kwargs) -> Dict[str, np.multiarray.ndarray]
        """Query the database for the values associated with the given parameters.

        All parameters must be specified.

        Parameters
        ----------
        **kwargs :
            parameter values.

        Returns
        -------
        results : Dict[str, np.ndarray]
            the characterization results.
        """
        fun_arg = self.get_fun_arg(**kwargs)

        results = {name: self.get_function(name, **kwargs)(fun_arg) for name in self._fun_names}

        for key in self._swp_names:
            results[key] = kwargs[key]

        return results
