# -*- coding: utf-8 -*-

"""This module contains design algorithm for a traditional two stage operational amplifier."""

from typing import List, Optional, Dict, Any

import scipy.optimize as sciopt

from ckt_dsn_ec.mos.core import MOSDBDiscrete

from .components import LoadDiodePFB, InputGm


class TailStage1(object):
    """Tail transistor of the first stage op amp.

    Due to layout restrictions, the tail transistor needs to have the same number of fingers
    and stack number as the input transistor.  This method finds the optimal width/intent.
    """
    def __init__(self, mos_db):
        # type: (MOSDBDiscrete) -> None
        self._db = mos_db
        self._intent_list = mos_db.get_dsn_param_values('intent')
        self._valid_widths = mos_db.width_list
        self._best_op = None

    def design(self,
               itarg_list,  # type: List[float]
               vd_list,  # type: List[float]
               vout_amp_list,  # type: List[float]
               vb,  # type: float
               l,  # type: float
               seg,  # type: int
               stack,  # type: int
               ):
        # type: (...) -> None

        vgs_idx = self._db.get_fun_arg_index('vgs')

        self._best_op = best_score = None
        for intent in self._intent_list:
            for w in self._valid_widths:
                self._db.set_dsn_params(l=l, w=w, intent=intent, stack=stack)
                ib = self._db.get_function_list('ibias')
                gds = self._db.get_function_list('gds')

                vgs_min, vgs_max = ib[0].get_input_range(vgs_idx)
                vg_min = vgs_min + vb
                vg_max = vgs_max + vb

                # find vgs for each corner
                vg_list, ro1_list, ro2_list = self._solve_vgs(itarg_list, vd_list, vout_amp_list, ib, gds, seg,
                                                              vb, vg_min, vg_max)
                if vg_list is not None:
                    cur_score = min(ro2_list)
                    if self._best_op is None or cur_score > best_score:
                        best_score = cur_score
                        self._best_op = (w, intent, vg_list, ro1_list, ro2_list)

    def _solve_vgs(self, itarg_list, vd_list, vout_amp_list, ib_list, gds_list, seg, vb, vg_min, vg_max):
        vg_list, ro1_list, ro2_list = [], [], []
        for itarg, vd, vo2, ibf, gdsf in zip(itarg_list, vd_list, vout_amp_list, ib_list, gds_list):

            def zero_fun(vg):
                farg = self._db.get_fun_arg(vbs=vb - vd, vds=vd - vb, vgs=vg - vb)
                return seg * ibf(farg) - itarg

            v1, v2 = zero_fun(vg_min), zero_fun(vg_max)
            if v1 < 0 and v2 < 0 or v1 > 0 and v2 > 0:
                # no solution
                return None, None, None

            vg_sol = sciopt.brentq(zero_fun, vg_min, vg_max)  # type: float
            arg1 = self._db.get_fun_arg(vbs=vb - vd, vds=vd - vb, vgs=vg_sol - vb)
            ro1 = 1 / gdsf(arg1)
            arg2 = self._db.get_fun_arg(vbs=vb - vd, vds=vo2 - vb, vgs=vg_sol - vb)
            ro2 = 1 / gdsf(arg2)
            vg_list.append(vg_sol)
            ro1_list.append(ro1)
            ro2_list.append(ro2)

        return vg_list, ro1_list, ro2_list

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        if self._best_op is None:
            return None

        w, intent, vg_list, ro1_list, ro2_list = self._best_op

        return dict(
            w=w,
            intent=intent,
            vg=vg_list,
            ro1=ro1_list,
            ro2=ro2_list
        )


class OpAmpTwoStage(object):
    """A two stage fully differential operational amplifier.

    The first stage is a differential amplifier with diode + positive feedback load, the
    second stage is a psuedo-differential common source amplifier.

    This topology has the following advantages:
    1. large output swing.
    2. Common mode feedback is only required for the second stage.
    """

    def __init__(self, nch_db, pch_db):
        # type: (MOSDBDiscrete, MOSDBDiscrete) -> None
        self._nch_db = nch_db
        self._pch_db = pch_db
        self._amp_info = None

    def design(self,
               itarg_list,  # type: List[float]
               vg_list,  # type: List[float]
               vout_list,  # type: List[float]
               l,  # type: float
               vstar_gm_min,  # type: float
               vstar_load_min,  # type: float
               vds_tail_min,  # type: float
               seg_gm_min,  # type: int
               vdd,  # type: float
               pmos_input=True,  # type: bool
               ):
        if pmos_input:
            load = LoadDiodePFB(self._nch_db)
            gm = InputGm(self._pch_db)
            tail1 = TailStage1(self._pch_db)
        else:
            load = LoadDiodePFB(self._pch_db)
            gm = InputGm(self._nch_db)
            tail1 = TailStage1(self._nch_db)

        # design load
        load.design(itarg_list, vstar_load_min, l)
        load_info = load.get_dsn_info()
        ro_load_list = load_info['ro']
        cdd_load_list = load_info['co']
        stack_ngm = load_info['stack_ngm']
        if pmos_input:
            vd_list = load_info['vgs']
            vb = vdd
        else:
            vd_list = [vdd - vgs for vgs in load_info['vgs']]
            vb = 0

        # design input gm
        gm.design(itarg_list, vg_list, vd_list, ro_load_list, vb, vstar_gm_min, vds_tail_min, l,
                  seg_min=seg_gm_min, stack_list=[stack_ngm])
        gm_info = gm.get_dsn_info()
        gm1_list = gm_info['gm']
        cdd_gm_list = gm_info['cdd']
        ro_gm_list = gm_info['ro']
        vtail_list = gm_info['vs']
        seg_gm = gm_info['seg']
        stack_gm = gm_info['stack']

        ro1_list = [1 / (1/ro_gm + 1/ro_load) for ro_gm, ro_load in zip(ro_gm_list, ro_load_list)]
        gain1_list = [gm1 * ro1 for gm1, ro1 in zip(gm1_list, ro1_list)]
        c1_list = [cd_l + cd_g for cd_l, cd_g in zip(cdd_load_list, cdd_gm_list)]

        # design stage 1 tail
        tail1.design(itarg_list, vtail_list, vout_list, vb, l, seg_gm, stack_gm)
        tail1_info = tail1.get_dsn_info()
        rn2_list = tail1_info['ro2']

        self._amp_info = dict(
            vtail=vtail_list,
            vmid=vd_list,
            vbias=tail1_info['vg'],

            vstar=gm_info['vstar'],
            cin=gm_info['cgg'],
            gm1=gm1_list,
            ro1=ro1_list,
            rt1=tail1_info['ro1'],
            gain1=gain1_list,
            c1=c1_list,

            w_tail1=tail1_info['w'],
            intent_tail1=tail1_info['intent'],

            w_gm=gm_info['w'],
            intent_gm=gm_info['intent'],
            seg_gm=seg_gm,
            stack_gm=stack_gm,

            w_load=load_info['w'],
            intent_load=load_info['intent'],
            seg_diode=load_info['seg_diode'],
            seg_ngm=load_info['seg_ngm'],
            stack_diode=load_info['stack_diode'],
            stack_ngm=load_info['stack_ngm'],
        )

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        return self._amp_info
