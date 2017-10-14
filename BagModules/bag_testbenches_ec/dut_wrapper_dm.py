# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

import os
import pkg_resources
from typing import Tuple, Sequence, Dict

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'dut_wrapper_dm.yaml'))


# noinspection PyPep8Naming
class bag_testbenches_ec__dut_wrapper_dm(Module):
    """A class that wraps a differential DUT to single-ended.
    """

    param_list = ['dut_lib', 'dut_cell', 'balun_list', 'pin_list', 'dut_conns']

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)
        for par in self.param_list:
            self.parameters[par] = None

    def design(self, dut_lib='', dut_cell='', balun_list=None, pin_list=None, dut_conns=None):
        # type: (str, str, Sequence[Tuple[str, str, str, str]], Sequence[Tuple[str, str]], Dict[str, str]) -> None
        """Design this wrapper schematic.

        This cell converts a variable number of differential pins to single-ended pins or
        vice-versa, by using ideal_baluns.  It can also create extra pins to be connected
        to the device.

        VDD and VSS pins will always be there for primary supplies.  Additional supplies
        can be added as inputOutput pins using the pin_list parameters.  If you don't need
        supply pins, they will be left unconnected.

        NOTE: the schematic template contains pins 'inac', 'indc', 'outac', and 'outdc' by
        default.  However, if they are not specified in pin_list, they will be deleted.
        In this way designer has full control over how they want the inputs/outputs to be
        named.

        Parameters
        ----------
        dut_lib : str
            DUT library name.
        dut_cell : str
            DUT cell name.
        balun_list: Sequence[Tuple[str, str, str, str]]
            list of balun connects to instantiate, represented as a list of
            (diff, comm, pos, neg) tuples.
        pin_list : Sequence[Tuple[str, str]]
            list of pins of this schematic, represented as a list of (name, purpose) tuples.
            purpose can be 'input', 'output', or 'inputOutput'.
        dut_conns : Dict[str, str]
            a dictionary from DUT pin name to the net name.  All connections should
            be specified, including VDD and VSS.
        """
        # error checking
        if not balun_list:
            raise ValueError('balun_list cannot be None or empty.')
        if not pin_list:
            raise ValueError('pin_list cannot be None or empty.')
        if not dut_conns:
            raise ValueError('dut_conns cannot be None or empty.')

        local_dict = locals()
        for name in self.param_list:
            if name not in local_dict:
                raise ValueError('Parameter %s not specified.' % name)
            self.parameters[name] = local_dict[name]

        # delete default input/output pins
        for pin_name in ('inac', 'indc', 'outac', 'outdc'):
            self.remove_pin(pin_name)

        # add pins
        for pin_name, pin_type in pin_list:
            self.add_pin(pin_name, pin_type)

        # replace DUT
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)

        # connect DUT
        for dut_pin, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', dut_pin, net_name)

        # add baluns and connect them
        num_balun = len(balun_list)
        name_list = ['XBAL%d' % idx for idx in range(num_balun)]
        self.array_instance('XBAL', name_list)
        for idx, (diff, comm, pos, neg) in enumerate(balun_list):
            self.reconnect_instance_terminal('XBAL', 'd', diff, index=idx)
            self.reconnect_instance_terminal('XBAL', 'c', comm, index=idx)
            self.reconnect_instance_terminal('XBAL', 'p', pos, index=idx)
            self.reconnect_instance_terminal('XBAL', 'n', neg, index=idx)
