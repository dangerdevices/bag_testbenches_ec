# -*- coding: utf-8 -*-

"""This module defines the AC testbench class."""

from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence

import numpy as np

from bag.simulation.core import TestbenchManager

if TYPE_CHECKING:
    from bag.core import Testbench


class ACTB(TestbenchManager):
    """This class sets up a generic AC analysis testbench.
    """
    def __init__(self,
                 data_fname,  # type: str
                 tb_name,  # type: str
                 impl_lib,  # type: str
                 specs,  # type: Dict[str, Any]
                 sim_view_list,  # type: Sequence[Tuple[str, str]]
                 env_list,  # type: Sequence[str]
                 ):
        # type: (...) -> None
        TestbenchManager.__init__(self, data_fname, tb_name, impl_lib, specs, sim_view_list, env_list)

    def setup_testbench(self, tb):
        # type: (Testbench) -> None
        fstart = self.specs['fstart']
        fstop = self.specs['fstop']
        fndec = self.specs['fndec']
        sim_vars = self.specs['sim_vars']
        sim_vars_env = self.specs.get('sim_vars_env', None)
        sim_outputs = self.specs.get('sim_outputs', None)

        tb.set_parameter('fstart', fstart)
        tb.set_parameter('fstop', fstop)
        tb.set_parameter('fndec', fndec)

        for key, val in sim_vars.items():
            if isinstance(val, int) or isinstance(val, float):
                tb.set_parameter(key, val)
            else:
                tb.set_sweep_parameter(key, values=val)

        if sim_vars_env is not None:
            for key, val in sim_vars_env.items():
                tb.set_env_parameter(key, val)

        if sim_outputs is not None:
            for key, val in sim_outputs.items():
                tb.add_output(key, val)

    def get_outputs(self):
        # type: () -> List[str]
        """Returns a list of output names."""
        sim_outputs = self.specs.get('sim_outputs', None)
        if sim_outputs is None:
            return []
        return list(sim_outputs.keys())

    def get_gain_and_w3db(self, f_vec, out_arr, swp_params):
        """Compute the DC gain and bandwidth of the amplifier given output.
        """
        # move frequency axis to last axis
        freq_idx = swp_params.index('freq')
        out_arr = np.moveaxis(out_arr, freq_idx, -1)
        gain_arr = out_arr[..., 0]

        out_log = 20 * np.log10(out_arr)
        gain_log_3db = 20 * np.log10(gain_arr) - 3


        # find first index at which gain goes below gain_log 3db
        diff_arr = out_log - gain_log_3db[..., np.newaxis]
        idx_arr = np.argmax(diff_arr < 0, axis=-1)
        freq_log = np.log10(f_vec)
        freq_log_max = freq_log[idx_arr]

        