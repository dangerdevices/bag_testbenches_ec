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
from typing import Union, List, Tuple, Any, Dict

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'mos_analogbase.yaml'))


# noinspection PyPep8Naming
class bag_testbenches_ec__mos_analogbase(Module):
    """Module for library bag_testbenches_ec cell mos_analogbase.

    Fill in high level description here.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            mos_type="Transistor type.  Either 'pch' or 'nch'.",
            w='Transistor width in meters or number of fins.',
            lch='Transistor length in meters.',
            fg='Transistor number of segments.',
            intent='Transistor threshold flavor.',
            stack='Number of stacked transistors in a segment.',
            dum_info='Dummy information data structure.',
        )

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        return dict(
            intent='standard',
            stack=1,
            dum_info=None,
        )

    def design(self,
               mos_type,  # type: str
               w,  # type: Union[float, int]
               lch,  # type: float
               fg,  # type: int
               intent,  # type: str
               stack,  # type: int
               dum_info,  # type: List[Tuple[Any]]
               ):
        # type: (...) -> None
        """Design a single transistor for characterization purposes.

        Parameters
        ----------
        mos_type : str
            the transistor type.  Either 'nch' or 'pch'.
        w : Union[float, int]
            transistor width, in fins or meters.
        lch : float
            transistor channel length, in meters.
        fg : int
            number of fingers.
        intent : str
            transistor threshold flavor.
        stack : int
            number of transistors in a stack.
        dum_info : List[Tuple[Any]]
            the dummy information data structure.
        """
        if fg == 1:
            raise ValueError('Cannot make 1 finger transistor.')
        # select the correct transistor type
        if mos_type == 'nch':
            self.delete_instance('XP')
            inst_name = 'XN'
        else:
            self.delete_instance('XN')
            inst_name = 'XP'

        if stack > 1:
            # array instances
            name_list = []
            term_list = []
            # add stack transistors
            for idx in range(stack):
                name_list.append('%s%d<%d:0>' % (inst_name, idx, fg - 1))
                cur_term = {}
                if idx != stack - 1:
                    cur_term['S'] = 'mid%d<%d:0>' % (idx, fg - 1)
                if idx != 0:
                    cur_term['D'] = 'mid%d<%d:0>' % (idx - 1, fg - 1)
                term_list.append(cur_term)

            # design transistors
            self.array_instance(inst_name, name_list, term_list=term_list)
            for idx in range(stack):
                self.instances[inst_name][idx].design(w=w, l=lch, nf=1, intent=intent)
        else:
            self.instances[inst_name].design(w=w, l=lch, nf=fg, intent=intent)

        # handle dummy transistors
        self.design_dummy_transistors(dum_info, 'XD', 'b', 'b')
