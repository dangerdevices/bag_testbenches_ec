# -*- coding: utf-8 -*-

"""This module contains various design methods/classes for amplifier components."""

from typing import List, Union, Tuple

import scipy.optimize as sciopt

from bag import float_to_si_string
from bag.util.search import BinaryIterator
from bag.math.dfun import DiffFunction

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
        self._intent_list = mos_db.get_dsn_param_values('intent')
        self._best_op = None

    def design(self, itarg_list, vstar_min, l, valid_width_list):
        # type: (List[float], float, float, List[Union[float, int]], float) -> None
        """Design the diode load.

        Parameters
        ----------
        itarg_list : List[float]
            target single-ended bias current across simulation environments.
        vstar_min : float
            minimum V* of the diode.
        l : float
            channel length.
        valid_width_list : List[Union[float, int]]
            list of valid width values.
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
        vgs_idx = self._db.get_fun_arg_index('vgs')

        num_stack = len(self._stack_list)

        best_op = None
        best_ro = None
        for intent in self._intent_list:
            for idx1 in range(num_stack):
                stack1 = self._stack_list[idx1]
                self._db.set_dsn_params(intent=intent, stack=stack1)
                ib1 = self._db.get_function_list('ibias')
                gm1 = self._db.get_function_list('gm')
                cgg1 = self._db.get_function_list('cgg')
                vgs1_min, vgs1_max = ib1[0].get_input_range(vgs_idx)

                for idx2 in range(idx1 + 1, num_stack):
                    stack2 = self._stack_list[idx2]
                    self._db.set_dsn_params(stack=stack2)
                    ib2 = self._db.get_function_list('ibias')
                    gm2 = self._db.get_function_list('gm')
                    cgg2 = self._db.get_function_list('cgg')
                    vgs2_min, vgs2_max = ib2[0].get_input_range(vgs_idx)

                    vgs_min = max(vgs1_min, vgs2_min)
                    vgs_max = min(vgs1_max, vgs2_max)

                    for w in valid_width_list:
                        scale = w / wnom
                        seg1_iter = BinaryIterator(2, None, step=2)
                        while seg1_iter.has_next():
                            seg1 = seg1_iter.get_next()
                            scale1 = scale * seg1

                            seg2_iter = BinaryIterator(seg1, None, step=2)
                            while seg2_iter.has_next():
                                seg2 = seg2_iter.get_next()
                                scale2 = scale * seg2

                                vgs_list, err_code = self._solve_vgs(itarg_list, scale1, scale2, ib1, ib2,
                                                                     vgs_min, vgs_max)
                                if err_code < 0:
                                    # too few fingers
                                    seg2_iter.up()
                                elif err_code > 0:
                                    # too many fingers
                                    seg2_iter.down()
                                else:
                                    good = True
                                    fun_args = [self._db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs) for vgs in vgs_list]
                                    ro_list, cg_list, vstar_list = [], [], []
                                    for fib1, fgm1, fgm2, fcg1, fcg2, arg in zip(ib1, gm1, gm2, cgg1, cgg2, fun_args):
                                        cur_gm1 = scale1 * fgm1(arg)
                                        cur_gm2 = scale2 * fgm2(arg)
                                        cur_cg = scale1 * fcg1(arg) + scale2 * fcg2(arg)
                                        cur_ib1 = scale1 * fib1(arg)
                                        cur_vstar = 2 * cur_ib1 / cur_gm1
                                        if cur_gm2 >= cur_gm1 or cur_vstar < vstar_min:
                                            # negative resistance or exceed max cap spec
                                            good = False
                                            break
                                        else:
                                            ro_list.append(1 / (cur_gm1 - cur_gm2))
                                            cg_list.append(cur_cg)
                                            vstar_list.append(cur_vstar)

                                    if good:
                                        seg2_iter.save()
                                        seg2_iter.up()
                                        cur_score = min(ro_list)
                                        if best_ro is None or cur_score > best_ro:
                                            best_ro = cur_score
                                            best_op = (intent, stack1, stack2, w, seg1, seg2,
                                                       vgs_list, ro_list, cg_list, vstar_list)
                                    else:
                                        seg2_iter.down()

                            if seg2_iter.get_last_save() is None:
                                # no solution for seg2, must broke cg_max spec
                                seg1_iter.down()
                            else:
                                seg1_iter.save()
                                seg1_iter.up()

        self._best_op = best_op

    def _solve_vgs(self,
                   itarg_list,  # type: List[float]
                   k1,  # type: float
                   k2,  # type: float
                   ib1,  # type: List[DiffFunction]
                   ib2,  # type: List[DiffFunction]
                   vgs_min,  # type: float
                   vgs_max,  # type: float
                   ):
        # type: (...) -> Tuple[List[float], int]

        vgs_list = []
        for itarg, ifun1, ifun2 in zip(itarg_list, ib1, ib2):
            def zero_fun(vgs):
                fun_arg = self._db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs)
                return ifun1(fun_arg) * k1 + ifun2(fun_arg) * k2 - itarg

            itest0 = zero_fun(vgs_min)
            itest1 = zero_fun(vgs_max)
            if itest0 < 0 and itest1 < 0:
                # too few fingers
                return [], -1
            elif itest0 > 0 and itest1 > 0:
                # too many fingers
                return [], 1
            else:
                vgs_cur = sciopt.brentq(zero_fun, vgs_min, vgs_max)
                vgs_list.append(vgs_cur)

        return vgs_list, 0

    def print_dsn_info(self):
        if self._best_op is None:
            print('No solution found.')
        else:
            intent, stack1, stack2, w, seg1, seg2, vgs_list, ro_list, cg_list, vstar_list = self._best_op

            print('intent = %s, stack1 = %d, stack2 = %d, w = %.2g, seg1 = %d, seg2 = %d' %
                  (intent, stack1, stack2, w, seg1, seg2))
            for name, val_list in (('vgs', vgs_list), ('ro', ro_list), ('cgg', cg_list), ('V*', vstar_list)):
                print('%s = [%s]' % (name, ', '.join(['%.3g' % val for val in val_list])))


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