# -*- coding: utf-8 -*-

"""This package contains query classes for transistor parameters."""


class MOSDBDiscrete(object):
    """Transistor small signal parameters database with discrete width choices.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    spec_list : List[str]
        list of specification file locations corresponding to widths.
    noise_fstart : Optional[float]
        noise integration frequency lower bound.  None to disable noise.
    noise_fstop : Optional[float]
        noise integration frequency upper bound.  None to disable noise.
    noise_scale : float
        noise integration scaling factor.
    noise_temp : float
        noise temperature.
    method : str
        interpolation method.
    cfit_method : str
        method used to fit capacitance to Y parameters.
    bag_config_path : Optional[str]
        BAG configuration file path.
    """

    def __init__(self,
                 spec_list,  # type: List[str]
                 noise_fstart=None,  # type: Optional[float]
                 noise_fstop=None,  # type: Optional[float]
                 noise_scale=1.0,  # type: float
                 noise_temp=300,  # type: float
                 method='linear',  # type: str
                 cfit_method='average',  # type: str
                 bag_config_path=None,  # type: Optional[str]
                 ):
        # type: (...) -> None
        # error checking

        tech_info = create_tech_info(bag_config_path=bag_config_path)

        self._width_res = tech_info.tech_params['mos']['width_resolution']
        self._sim_envs = None
        self._ss_swp_names = None
        self._sim_list = []
        self._ss_list = []
        self._ss_outputs = None
        self._width_list = []
        for spec in spec_list:
            sim = MOSCharSS(None, spec)
            self._width_list.append(int(round(sim.width / self._width_res)))
            # error checking
            if 'w' in sim.swp_var_list:
                raise ValueError('MOSDBDiscrete assumes transistor width is not swept.')

            corners, ss_swp_names, ss_dict = sim.get_ss_info(noise_fstart, noise_fstop,
                                                             scale=noise_scale, temp=noise_temp,
                                                             method=method, cfit_method=cfit_method)
            if self._sim_envs is None:
                self._ss_swp_names = ss_swp_names
                self._sim_envs = corners
                test_dict = next(iter(ss_dict.values()))
                self._ss_outputs = sorted(test_dict.keys())
            elif self._sim_envs != corners:
                raise ValueError('Simulation environments mismatch between given specs.')
            elif self._ss_swp_names != ss_swp_names:
                raise ValueError('signal-signal parameter sweep names mismatch.')

            self._sim_list.append(sim)
            self._ss_list.append(ss_dict)

        self._env_list = self._sim_envs
        self._cur_idx = 0
        self._dsn_params = dict(w=self._width_list[0])

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
        w_unit = int(round(self._dsn_params['w'] / self._width_res))
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
