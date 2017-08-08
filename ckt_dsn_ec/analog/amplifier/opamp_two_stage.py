# -*- coding: utf-8 -*-

"""This module contains design algorithm for a traditional two stage operational amplifier."""

from typing import List, Optional, Dict, Any

import numpy as np
import scipy.optimize as sciopt

from bag.math import gcd
from bag.data.lti import LTICircuit, get_stability_margins, get_w_crossings
from bag.util.search import FloatBinaryIterator

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
                        self._best_op = (w, intent, seg, stack, vb, vg_list, vout_amp_list, ro1_list, ro2_list)

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
            ro1 = 1 / float(gdsf(arg1))
            arg2 = self._db.get_fun_arg(vbs=vb - vd, vds=vo2 - vb, vgs=vg_sol - vb)
            ro2 = 1 / float(gdsf(arg2))
            vg_list.append(vg_sol)
            ro1_list.append(ro1)
            ro2_list.append(ro2)

        return vg_list, ro1_list, ro2_list

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        if self._best_op is None:
            return None

        w, intent, seg, stack, vb, vg_list, vout_list, ro1_list, ro2_list = self._best_op
        self._db.set_dsn_params(w=w, intent=intent, stack=stack)
        cdd = self._db.get_function_list('cdd')
        cdd_list = []
        for vg, vout, cddf in zip(vg_list, vout_list, cdd):
            arg = self._db.get_fun_arg(vbs=0, vds=vout - vb, vgs=vg - vb)
            cur_cdd = cddf(arg)  # type: float
            cur_cdd = seg * float(cur_cdd)
            cdd_list.append(cur_cdd)

        return dict(
            w=w,
            intent=intent,
            vg=vg_list,
            ro1=ro1_list,
            ro2=ro2_list,
            cdd=cdd_list,
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
               cload,  # type: float
               f_unit,  # type: float
               phase_margin,  # type: float
               res_var,  # type: float
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
        gm2_list = load_info['gm']
        gds2_list = load_info['gds']
        ro_load_list = load_info['ro']
        cgg_load_list = load_info['cgg']
        cdd_load_list = load_info['cdd']
        stack_diode = load_info['stack_diode']
        stack_ngm = load_info['stack_ngm']
        seg_diode = load_info['seg_diode']
        seg_ngm = load_info['seg_ngm']
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

        ro1_list = [1 / (1 / ro_gm + 1 / ro_load) for ro_gm, ro_load in zip(ro_gm_list, ro_load_list)]
        gain1_list = [gm1 * ro1 for gm1, ro1 in zip(gm1_list, ro1_list)]
        c1_list = [cd_l + cd_g for cd_l, cd_g in zip(cgg_load_list, cdd_gm_list)]

        # design stage 1 tail
        tail1.design(itarg_list, vtail_list, vout_list, vb, l, seg_gm, stack_gm)
        tail1_info = tail1.get_dsn_info()
        rn2_list = tail1_info['ro2']
        cdd_tail_list = tail1_info['cdd']

        # design stage 2 gm
        ro2_list = [1/(gds_l + 1 / ro_t) for gds_l, ro_t in zip(gds2_list, rn2_list)]
        c2_list = [cd_l + cd_t for cd_l, cd_t in zip(cdd_load_list, cdd_tail_list)]
        self.design_stage2(gm1_list, gm2_list, ro1_list, ro2_list, c1_list, c2_list,
                           cload, res_var, phase_margin, f_unit, seg_gm, seg_diode, seg_ngm)

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
            gm2=gm2_list,

            w_tail1=tail1_info['w'],
            intent_tail1=tail1_info['intent'],

            w_gm=gm_info['w'],
            intent_gm=gm_info['intent'],
            seg_gm=seg_gm,
            stack_gm=stack_gm,

            w_load=load_info['w'],
            intent_load=load_info['intent'],
            seg_diode=seg_diode,
            seg_ngm=seg_ngm,
            stack_diode=stack_diode,
            stack_ngm=stack_ngm,
        )

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        return self._amp_info

    def design_stage2(self, gm1_list, gm2_list, ro1_list, ro2_list, c1_list, c2_list,
                      cload, res_var, phase_margin, f_unit, seg_tail, seg_diode, seg_ngm):
        # step 1: find stage 2 unit size
        f = gcd(gcd(seg_tail, seg_diode), seg_ngm)
        if f % 2 != 0:
            raise ValueError('All segment numbers must be even.')
        f //= 2
        cur_size = 1

        found = False
        while not found:
            cur_gm2_list, cur_ro2_list, cur_c2_list = [], [], []
            for gm2, ro2, c2 in zip(gm2_list, ro2_list, c2_list):
                cur_gm2_list.append(gm2 * cur_size / f)
                cur_ro2_list.append(ro2 * f / cur_size)
                cur_c2_list.append(cload + c2 * cur_size / f)

            rz, cf, gain_list, bw_list, pm_list = self._find_rz_cf(gm1_list, cur_gm2_list, ro1_list, cur_ro2_list,
                                                                   c1_list, cur_c2_list, res_var, phase_margin)

            print('cur_scale = %d / %d' % (cur_size, f))
            print('gain: [%s]' % (', '.join(('%.3g' % v for v in gain_list))))
            print('bw: [%s]' % (', '.join(('%.3g' % v for v in bw_list))))
            print('pm: [%s]' % (', '.join(('%.3g' % v for v in pm_list))))
            if min(bw_list) > f_unit:
                found = True
            else:
                cur_size += 1

    @classmethod
    def _make_circuit(cls, gm1, gm2, r1, r2, c1, c2, rz):
        cir = LTICircuit()
        cir.add_res(r1, 'vx', 'gnd')
        cir.add_cap(c1, 'vx', 'gnd')
        cir.add_vccs(gm1, 'vx', 'gnd', 'vi')
        cir.add_res(r2, 'vo', 'gnd')
        cir.add_cap(c2, 'vo', 'gnd')
        cir.add_vccs(gm2, 'vo', 'gnd', 'vx')
        cir.add_res(rz, 'vx', 'vm')
        return cir

    def _find_rz_cf(self, gm1_list, gm2_list, r1_list, r2_list, c1_list, c2_list, res_var, phase_margin,
                    cap_tol=1e-15, cap_step=10e-15):
        """Find minimum miller cap that stabilizes the system.

        NOTE: This function assume phase of system for any miller cap value will not loop around 360,
        otherwise it may get the phase margin wrong.  This assumption should be valid for this op amp.
        """
        rz_worst = 1 / min(gm2_list)
        rz_nom = rz_worst / (1 - res_var)
        # find maximum Cf needed to stabilize all corners
        cf_min = cap_step
        for gm1, gm2, r1, r2, c1, c2 in zip(gm1_list, gm2_list, r1_list, r2_list, c1_list, c2_list):
            cir = self._make_circuit(gm1, gm2, r1, r2, c1, c2, rz_worst)

            bin_iter = FloatBinaryIterator(cf_min, None, cap_tol, search_step=cap_step)
            while bin_iter.has_next():
                cur_cf = bin_iter.get_next()
                cir.add_cap(cur_cf, 'vm', 'vo')
                num, den = cir.get_num_den('vi', 'vo')
                cur_pm, _ = get_stability_margins(num, den)
                if cur_pm < phase_margin:
                    bin_iter.up()
                else:
                    bin_iter.save()
                    bin_iter.down()
                cir.add_cap(-cur_cf, 'vm', 'vo')

            cf_min = bin_iter.get_last_save()

        # find gain, unity gain bandwidth, and phase margin across corners
        gain_list, bw_list, pm_list = [], [], []
        for gm1, gm2, r1, r2, c1, c2 in zip(gm1_list, gm2_list, r1_list, r2_list, c1_list, c2_list):
            cir = self._make_circuit(gm1, gm2, r1, r2, c1, c2, rz_worst)
            cir.add_cap(cf_min, 'vm', 'vo')
            num, den = cir.get_num_den('vi', 'vo')
            pn = np.poly1d(num)
            pd = np.poly1d(num)
            gain_list.append(abs(pn(0) / pd(0)))
            bw_list.append(get_w_crossings(num, den)[0] / 2 / np.pi)
            pm_list.append(get_stability_margins(num, den)[0])

        return rz_nom, cf_min, gain_list, bw_list, pm_list
