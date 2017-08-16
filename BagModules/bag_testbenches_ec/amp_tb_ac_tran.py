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

from bag import float_to_si_string
from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'amp_tb_ac_tran.yaml'))


# noinspection PyPep8Naming
class bag_testbenches_ec__amp_tb_ac_tran(Module):
    """Module for library bag_testbenches_ec cell amp_tb_ac_tran.

    Fill in high level description here.
    """

    param_list = ['dut_lib', 'dut_cell', 'dut_conns', 'vbias_dict', 'ibias_dict', 'tran_fname', 'no_cload']

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)
        for par in self.param_list:
            self.parameters[par] = None

    def _generate_biases(self, bias_dict, name_template, inst_name, param_name):
        names = sorted(bias_dict.keys())
        name_list, term_list, val_list = [], [], []
        for name in names:
            pname, nname, bias_val = bias_dict[name]
            term_list.append(dict(PLUS=pname, MINUS=nname))
            name_list.append(name_template % name)
            if isinstance(bias_val, str):
                val_list.append(bias_val)
            else:
                val_list.append(float_to_si_string(bias_val))

        self.array_instance(inst_name, name_list, term_list=term_list)
        for inst, val in zip(self.instances[inst_name], val_list):
            inst.parameters[param_name] = val

    def design(self, dut_lib='', dut_cell='', dut_conns=None,
               vbias_dict=None, ibias_dict=None, tran_fname='', no_cload=False):
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        if vbias_dict is None:
            vbias_dict = {}
        if ibias_dict is None:
            ibias_dict = {}
        if dut_conns is None:
            dut_conns = {}

        local_dict = locals()
        for name in self.param_list:
            if name not in local_dict:
                raise ValueError('Parameter %s not specified.' % name)
            self.parameters[name] = local_dict[name]

        # setup bias voltages
        if vbias_dict:
            # make sure VDD is always included
            name = 'SUP'
            counter = 1
            while name in vbias_dict:
                name = 'SUP%d' % counter
                counter += 1

            vbias_dict[name] = ['VDD', 'VSS', 'vdd']
            self._generate_biases(vbias_dict, 'V%s', 'VSUP', 'vdc')

        # setup bias currents
        if not ibias_dict:
            self.delete_instance('IBIAS')
        else:
            self._generate_biases(ibias_dict, 'I%s', 'IBIAS', 'idc')

        # setup pwl source
        self.instances['VIN'].parameters['fileName'] = tran_fname

        # delete load cap if needed
        if no_cload:
            self.delete_instance('CLOAD')

        # setup DUT
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term_name, net_name)

    def get_layout_params(self, **kwargs):
        """Returns a dictionary with layout parameters.

        This method computes the layout parameters used to generate implementation's
        layout.  Subclasses should override this method if you need to run post-extraction
        layout.

        Parameters
        ----------
        kwargs :
            any extra parameters you need to generate the layout parameters dictionary.
            Usually you specify layout-specific parameters here, like metal layers of
            input/output, customizable wire sizes, and so on.

        Returns
        -------
        params : dict[str, any]
            the layout parameters dictionary.
        """
        return {}

    def get_layout_pin_mapping(self):
        """Returns the layout pin mapping dictionary.

        This method returns a dictionary used to rename the layout pins, in case they are different
        than the schematic pins.

        Returns
        -------
        pin_mapping : dict[str, str]
            a dictionary from layout pin names to schematic pin names.
        """
        return {}
