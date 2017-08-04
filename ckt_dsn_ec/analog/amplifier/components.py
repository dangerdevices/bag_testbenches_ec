# -*- coding: utf-8 -*-

"""This module contains various design methods/classes for amplifier components."""

from typing import List, Union, Tuple, Optional, Dict, Any

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
        # type: (List[float], float, float, List[Union[float, int]]) -> None
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

        self._best_op = None
        best_score = None
        for intent in self._intent_list:
            for idx1 in range(num_stack):
                stack1 = self._stack_list[idx1]
                self._db.set_dsn_params(intent=intent, stack=stack1)
                ib1 = self._db.get_function_list('ibias')
                gm1 = self._db.get_function_list('gm')
                vgs1_min, vgs1_max = ib1[0].get_input_range(vgs_idx)

                for idx2 in range(idx1 + 1, num_stack):
                    stack2 = self._stack_list[idx2]
                    self._db.set_dsn_params(stack=stack2)
                    ib2 = self._db.get_function_list('ibias')
                    gm2 = self._db.get_function_list('gm')
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
                                    cur_score = self._compute_score(vstar_min, scale1, scale2, ib1, gm1, gm2, vgs_list)

                                    if cur_score is None:
                                        seg2_iter.down()
                                    else:
                                        seg2_iter.save()
                                        seg2_iter.up()
                                        if best_score is None or cur_score > best_score:
                                            best_score = cur_score
                                            self._best_op = (intent, stack1, stack2, w, seg1, seg2, vgs_list)

                            if seg2_iter.get_last_save() is None:
                                # no solution for seg2, must broke cg_max spec
                                seg1_iter.down()
                            else:
                                seg1_iter.save()
                                seg1_iter.up()

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

    def _compute_score(self, vstar_min, scale1, scale2, ib1, gm1, gm2, vgs_list):
        score = float('inf')
        for fib1, fgm1, fgm2, vgs in zip(ib1, gm1, gm2, vgs_list):
            arg = self._db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs)
            cur_gm1 = scale1 * fgm1(arg)
            cur_gm2 = scale2 * fgm2(arg)
            cur_ib1 = scale1 * fib1(arg)
            cur_vstar = 2 * cur_ib1 / cur_gm1
            if cur_gm2 >= cur_gm1 or cur_vstar < vstar_min:
                return None

            score = min(score, 1 / (cur_gm1 - cur_gm2))

        return score

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        if self._best_op is None:
            return None

        intent, stack1, stack2, w, seg1, seg2, vgs_list = self._best_op
        wnom = self._db.get_default_dsn_value('w')

        self._db.set_dsn_params(intent=intent, stack=stack1)
        ib1 = self._db.get_function_list('ibias')
        gm1 = self._db.get_function_list('gm')
        cg1 = self._db.get_function_list('cgg')
        self._db.set_dsn_params(intent=intent, stack=stack2)
        gm2 = self._db.get_function_list('gm')
        cg2 = self._db.get_function_list('cgg')

        k1 = w * seg1 / wnom
        k2 = w * seg2 / wnom
        vstar_list, ro_list, co_list = [], [], []
        for ib1f, gm1f, cg1f, gm2f, cg2f, vgs in zip(ib1, gm1, cg1, gm2, cg2, vgs_list):
            arg = self._db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs)
            cur_ib1 = k1 * ib1f(arg)
            cur_gm1 = k1 * gm1f(arg)
            cur_gm2 = k2 * gm2f(arg)
            cur_cg1 = k1 * cg1f(arg)
            cur_cg2 = k2 * cg2f(arg)
            vstar_list.append(2 * cur_ib1 / cur_gm1)
            ro_list.append(1 / (cur_gm1 - cur_gm2))
            co_list.append(cur_cg1 + cur_cg2)

        return dict(
            vgs=vgs_list,
            vstar=vstar_list,
            ro=ro_list,
            co=co_list,
            intent=intent,
            stack1=stack1,
            stack2=stack2,
            w=w,
            seg1=seg1,
            seg2=seg2,
        )


class InputGm(object):
    """A simple differential input gm stage.

    This class maximizes the gain given V* constraint.
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

    def design(self,
               itarg_list,  # type: List[float]
               vg_list,  # type: List[float]
               vd_list,  # type: List[float]
               rload_list,  # type: List[float]
               vb,  # type: float
               vstar_min,  # type: float
               l,  # type: float
               valid_width_list,  # type: List[Union[float, int]]
               ):
        # type: (...) -> None
        """Design the input gm stage.

        Parameters
        ----------
        itarg_list : List[float]
            target single-ended bias current across simulation environments.
        vg_list : List[float]
            gate voltage across simulation environments.
        vd_list : List[float]
            drain voltage across simulation environments.
        rload_list : List[float]
            load resistance across simulation environments.
        vb : float
            body bias voltage.
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
        vds_idx = self._db.get_fun_arg_index('vds')

        best_score = None
        best_op = None
        for intent in self._intent_list:
            for stack in self._stack_list:
                self._db.set_dsn_params(intent=intent, stack=stack)
                ib = self._db.get_function_list('ibias')
                gm = self._db.get_function_list('gm')
                gds = self._db.get_function_list('gds')
                iunit_list = self._solve_iunit_from_vstar(vstar_min, vg_list, vd_list, ib, gm)
                tot_wunit = wnom * min((itarg / iunit for itarg, iunit in zip(itarg_list, iunit_list)))
                # now get actual numbers
                for w in valid_width_list:
                    num_seg = int(tot_wunit / w // 2) * 2
                    scale = w * num_seg / wnom
                    vs_list = self._solve_vs(itarg_list, vg_list, vd_list, scale, ib)

                    for gm_fun, gds_fun, rload, vg, vd, vs in zip(gm, gds, rload_list, vg_list, vd_list, vs_list):
                        arg = self._db.get_fun_arg(vbs=vb - vs, vds=vd - vs, vgs=vg - vs)
                        cur_gm = gm_fun(arg)
                        cur_gds = gds_fun(arg)
                        cur_gain = cur_gm / (1 / rload + cur_gds)
                        if best_score is None or cur_gain > best_score:
                            best_score = cur_gain
                            best_op = (intent, stack, w, num_seg, vs_list)




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