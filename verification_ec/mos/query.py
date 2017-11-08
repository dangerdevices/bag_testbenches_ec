# -*- coding: utf-8 -*-

"""This package contains query classes for transistor parameters."""

from typing import TYPE_CHECKING, List, Optional, Union, Sequence, Tuple, Any, Dict

import os

import numpy as np

from bag.core import create_tech_info
from bag.simulation.core import DesignManager
from bag.io.sim_data import load_sim_file
from bag.math.interpolate import interpolate_grid
from bag.math.dfun import VectorDiffFunction

if TYPE_CHECKING:
    from bag.math.dfun import DiffFunction


class MOSDBDiscrete(object):
    """Transistor small signal parameters database with discrete width choices.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    spec_list : List[str]
        list of specification file locations corresponding to widths.
    interp_method : str
        interpolation method.
    bag_config_path : Optional[str]
        BAG configuration file path.
    meas_type : str
        transistor characterization measurement type.
    """

    def __init__(self,
                 spec_list,  # type: List[str]
                 interp_method='spline',  # type: str
                 bag_config_path=None,  # type: Optional[str]
                 meas_type='mos_ss',  # type: str
                 ):
        # type: (...) -> None
        # error checking

        tech_info = create_tech_info(bag_config_path=bag_config_path)

        self._width_res = tech_info.tech_params['mos']['width_resolution']
        self._sim_envs = None
        self._ss_swp_names = None
        self._manager_list = []  # type: List[DesignManager]
        self._ss_list = []
        self._ss_outputs = None
        self._width_list = []

        for spec in spec_list:
            dsn_manager = DesignManager(None, spec)
            cur_width = dsn_manager.specs['layout_params']['w']
            cur_width = int(round(cur_width / self._width_res))
            self._width_list.append(cur_width)

            # error checking
            if 'w' in dsn_manager.swp_var_list:
                raise ValueError('MOSDBDiscrete assumes transistor width is not swept.')

            ss_fun_table = {}
            for dsn_name in dsn_manager.get_dsn_name_iter():
                meas_dir = dsn_manager.get_measurement_directory(dsn_name, meas_type)
                ss_dict = load_sim_file(os.path.join(meas_dir, 'ss_params.hdf5'))

                cur_corners = ss_dict['corner'].tolist()
                cur_ss_swp_names = ss_dict['sweep_params']['ibias'][1:]
                if self._sim_envs is None:
                    # assign attributes for the first time
                    self._sim_envs = cur_corners
                    self._ss_swp_names = cur_ss_swp_names
                elif self._sim_envs != cur_corners:
                    raise ValueError('Simulation environments mismatch between given specs.')
                elif self._ss_swp_names != cur_ss_swp_names:
                    raise ValueError('signal-signal parameter sweep names mismatch.')

                cur_fun_dict = self._make_ss_functions(ss_dict, cur_corners, cur_ss_swp_names, interp_method)

                if self._ss_outputs is None:
                    self._ss_outputs = sorted(cur_fun_dict.keys())

                ss_fun_table[dsn_name] = cur_fun_dict

            self._manager_list.append(dsn_manager)
            self._ss_list.append(ss_fun_table)

        self._env_list = self._sim_envs
        self._cur_idx = 0
        self._dsn_params = dict(w=self._width_list[0])

    @classmethod
    def _make_ss_functions(cls, ss_dict, corners, swp_names, interp_method):
        scale_list = []
        for name in swp_names:
            cur_xvec = ss_dict[name]
            scale_list.append((cur_xvec[0], cur_xvec[1] - cur_xvec[0]))

        fun_table = {}
        corner_sort_arg = np.argsort(corners)  # type: Sequence[int]
        for key in ss_dict['sweep_params'].keys():
            arr = ss_dict[key]
            fun_list = []
            for idx in corner_sort_arg:
                fun_list.append(interpolate_grid(scale_list, arr[idx, ...], method=interp_method,
                                                 extrapolate=True, delta=1e-5))
            fun_table[key] = fun_list

        # add derived parameters
        cgdl = fun_table['cgd']
        cgsl = fun_table['cgs']
        cgbl = fun_table['cgb']
        cdsl = fun_table['cds']
        cdbl = fun_table['cdb']
        csbl = fun_table['csb']
        ss_dict['cgg'] = [cgd + cgs + cgb for (cgd, cgs, cgb) in zip(cgdl, cgsl, cgbl)]
        ss_dict['cdd'] = [cgd + cds + cdb for (cgd, cds, cdb) in zip(cgdl, cdsl, cdbl)]
        ss_dict['css'] = [cgs + cds + csb for (cgs, cds, csb) in zip(cgsl, cdsl, csbl)]

        return fun_table

    @property
    def width_list(self):
        # type: () -> List[Union[float, int]]
        """Returns the list of widths in this database."""
        return [w * self._width_res for w in self._width_list]

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
        return self._manager_list[self._cur_idx].swp_var_list

    def get_dsn_param_values(self, var):
        # type: (str) -> List[Any]
        """Returns a list of valid design parameter values."""
        return self._manager_list[self._cur_idx].get_swp_var_values(var)

    def set_dsn_params(self, **kwargs):
        # type: (**kwargs) -> None
        """Set the design parameters for which this database will query for."""
        self._dsn_params.update(kwargs)
        w_unit = int(round(self._dsn_params['w'] / self._width_res))
        self._cur_idx = self._width_list.index(w_unit)

    def _get_dsn_name(self, **kwargs):
        # type: (**kwargs) -> str
        if kwargs:
            self.set_dsn_params(**kwargs)

        combo_list = tuple(self._dsn_params[var] for var in self.dsn_params)
        dsn_name = self._manager_list[self._cur_idx].get_design_name(combo_list)
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
        if not env and len(self.env_list) == 1:
            env = self.env_list[0]

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

        return self._ss_swp_names, sample_fun.input_ranges

    def get_fun_arg(self, **kwargs):
        # type: (**kwargs) -> np.multiarray.ndarray
        """Convert keyword arguments to function argument."""
        return np.array([kwargs[key] for key in self._ss_swp_names])

    def get_fun_arg_index(self, name):
        # type: (str) -> int
        """Returns the function input argument index for the given variable"""
        return self._ss_swp_names.index(name)

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
        results = {name: self.get_function(name, **kwargs)(fun_arg) for name in self._ss_outputs}

        for key in self._ss_swp_names:
            results[key] = kwargs[key]

        return results
