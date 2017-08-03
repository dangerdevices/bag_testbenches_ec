# -*- coding: utf-8 -*-

"""This module contains various design methods/classes for amplifier components."""

from typing import List, Union

from bag import float_to_si_string

from ckt_dsn_ec.mos.core import MOSDB


class LoadDiodePFB(object):
    """A differential load consists of diode transistor and negative gm cell.

    This topology is designed to have a large differential mode resistance and a
    small common mode resistance, plus a well defined output common mode

    Parameters
    ----------
    mos_db : MOSDB
        the transistor small signal parameters database.
    """
    def __init__(self, mos_db):
        # type: (MOSDB) -> None
        self._db = mos_db
        self._dsn_params = mos_db.dsn_params
        if 'w' in self._dsn_params:
            raise ValueError('This class assumes transistor width is not swept.')
        if 'stack' not in self._dsn_params:
            raise ValueError('This class assumes transistor stack is swept.')

        self._stack_list = sorted(mos_db.get_dsn_param_values('stack'))

    def design(self, ibias, cg_max, l, valid_width_list, vgs_res=5e-3):
        # type: (float, float, float, List[Union[float, int]], float) -> None
        """Design the diode load.

        Parameters
        ----------
        ibias : float
            single-ended bias current.
        cg_max : float
            maximum single-ended gate capacitance.
        l : float
            channel length.
        valid_width_list : List[Union[float, int]]
            list of valid width values.
        vgs_res : float
            vgs resolution.
        """
        # simple error checking.
        if 'l' in self._dsn_params:
            self._db.set_dsn_params(l=l)
        else:
            lstr = float_to_si_string(l)
            db_lstr = float_to_si_string(self._db.get_default_dsn_value('l'))
            if lstr != db_lstr:
                raise ValueError('Given length = %s, but DB length = %s' % (lstr, db_lstr))

        wnom = self._db.get_default_dsn_value('w')

        num_stack = len(self._stack_list)
        for idx_diode in range(num_stack):
            stack_diode = self._stack_list[idx_diode]
            self._db.set_dsn_params(stack=stack_diode)
            ibias_diode = self._db.get_function('ibias')
            gm_diode = self._db.get_function('gm')
            cgg_diode = self._db.get_function('cgg')
            for idx_neg in range(idx_diode + 1, num_stack):
                stack_neg = self._stack_list[idx_neg]
                self._db.set_dsn_params(stack=stack_diode)
                ibias_neg = self._db.get_function('ibias')
                gm_neg = self._db.get_function('gm')
                cgg_neg = self._db.get_function('cgg')



"""
PMOS load:

given current, cg_max, length, get:

w
intent
stack1
stack2

fg1
fg2
rod
roc
vgs

that maximize output resistance.

Gm input stage with cascode:

given current, vg, vd, rop, length get:

w1
w2
intent1
intent2
stack1
stack2

fg1
fg2

vs
gm
ron

that maximize gm * (ron || rop)

second stage amp load:

given current, length, vo, cd_max, get:

wn
intent
stack

fg
vgs
ro

that maximize ro subject to cap constraint.

second stage transistor:

given current, length, wp, intentp, ron, get:

stackp

fg

that maximize gmp * (rop || ron)



"""