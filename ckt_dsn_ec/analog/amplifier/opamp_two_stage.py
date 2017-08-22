# -*- coding: utf-8 -*-

"""This module contains design algorithm for a traditional two stage operational amplifier."""

from typing import List, Optional, Dict, Any

import numpy as np
import scipy.optimize as sciopt

from bag.math import gcd
from bag.data.lti import LTICircuit, get_stability_margins, get_w_crossings, get_w_3db
from bag.util.search import FloatBinaryIterator, BinaryIterator

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
                vgs_list, gds1_list, gds2_list = self._solve_vgs(itarg_list, vout_amp_list, vd_list, ib, gds, seg,
                                                                 vb, vg_min, vg_max)
                if vgs_list is not None:
                    cur_score = max(gds2_list)
                    if self._best_op is None or cur_score < best_score:
                        best_score = cur_score
                        self._best_op = (w, intent, seg, stack, vb, vgs_list, vout_amp_list, gds1_list, gds2_list)

    def _solve_vgs(self, itarg_list, vout_list, vd_list, ib_list, gds_list, seg, vb, vg_min, vg_max):
        vgs_list, gds1_list, gds2_list = [], [], []
        for itarg, vout, vd, ibf, gdsf in zip(itarg_list, vout_list, vd_list, ib_list, gds_list):

            def zero_fun(vg):
                farg = self._db.get_fun_arg(vbs=vb - vd, vds=vd - vb, vgs=vg - vb)
                return seg * ibf(farg) - itarg

            v1, v2 = zero_fun(vg_min), zero_fun(vg_max)
            if v1 < 0 and v2 < 0 or v1 > 0 and v2 > 0:
                # no solution
                return None, None, None

            vg_sol = sciopt.brentq(zero_fun, vg_min, vg_max)  # type: float
            vgs_opt = vg_sol - vb
            arg1 = self._db.get_fun_arg(vbs=vb - vd, vds=vd - vb, vgs=vgs_opt)
            arg2 = self._db.get_fun_arg(vbs=vb - vd, vds=vout - vb, vgs=vgs_opt)
            vgs_list.append(vgs_opt)
            gds1_list.append(seg * gdsf(arg1))
            gds2_list.append(seg * gdsf(arg2))

        return vgs_list, gds1_list, gds2_list

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        if self._best_op is None:
            return None

        w, intent, seg, stack, vb, vgs_list, vout_list, gds1_list, gds2_list = self._best_op
        self._db.set_dsn_params(w=w, intent=intent, stack=stack)
        cdd = self._db.get_function_list('cdd')
        cdd2_list = []
        for vgs, vout, cddf in zip(vgs_list, vout_list, cdd):
            arg = self._db.get_fun_arg(vbs=0, vds=vout - vb, vgs=vgs)
            cur_cdd = cddf(arg)  # type: float
            cdd2_list.append(seg * cur_cdd)

        return dict(
            w=w,
            intent=intent,
            vgs=vgs_list,
            gds1=gds1_list,
            gds2=gds2_list,
            cdd2=cdd2_list,
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
               cpar1,  # type: float
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
               max_ref_ratio=20,  # type: int
               ):
        if pmos_input:
            load = LoadDiodePFB(self._nch_db)
            gm = InputGm(self._pch_db)
            tail1 = TailStage1(self._pch_db)
            vds2_list = vout_list
        else:
            load = LoadDiodePFB(self._pch_db)
            gm = InputGm(self._nch_db)
            tail1 = TailStage1(self._nch_db)
            vds2_list = [vo - vdd for vo in vout_list]

        # design load
        load.design(itarg_list, vds2_list, vstar_load_min, l)
        load_info = load.get_dsn_info()
        vgs_load_list = load_info['vgs']
        gds_load_list = load_info['gds1']
        ctot_load_list = load_info['ctot1']
        gm2_list = load_info['gm2']
        gds_in2_list = load_info['gds2']
        cgg_in2_list = load_info['cgg2']
        cdd_in2_list = load_info['cdd2']
        stack_diode = load_info['stack_diode']
        stack_ngm = load_info['stack_ngm']
        seg_diode = load_info['seg_diode']
        seg_ngm = load_info['seg_ngm']
        if pmos_input:
            vd_list = vgs_load_list
            vb = vdd
        else:
            vd_list = [vdd - vgs for vgs in vgs_load_list]
            vb = 0

        # design input gm
        gm.design(itarg_list, vg_list, vd_list, gds_load_list, vb, vstar_gm_min, vds_tail_min, l,
                  seg_min=seg_gm_min, stack_list=[stack_ngm])
        gm_info = gm.get_dsn_info()
        gm1_list = gm_info['gm']
        cdd_in_list = gm_info['cdd']
        gds_in_list = gm_info['gds']
        vtail_list = gm_info['vs']
        seg_gm = gm_info['seg']
        stack_gm = gm_info['stack']

        gds1_list = [gds_in + gds_load for gds_in, gds_load in zip(gds_in_list, gds_load_list)]
        gain1_list = [gm1 / gds1 for gm1, gds1 in zip(gm1_list, gds1_list)]
        c1_list = [ctot_l + cd_i for ctot_l, cd_i in zip(ctot_load_list, cdd_in_list)]

        # design stage 1 tail
        tail1.design(itarg_list, vtail_list, vout_list, vb, l, seg_gm, stack_gm)
        tail1_info = tail1.get_dsn_info()
        gds_tail2_list = tail1_info['gds2']
        cdd_tail2_list = tail1_info['cdd2']
        vbias_list = [vgs_tail + vb for vgs_tail in tail1_info['vgs']]

        # design stage 2 gm
        gds2_list = [gds_t2 + gds_i2 for gds_t2, gds_i2 in zip(gds_tail2_list, gds_in2_list)]
        c2_list = [cdd_t2 + cdd_i2 for cdd_t2, cdd_i2 in zip(cdd_tail2_list, cdd_in2_list)]
        stage2_results = self.design_stage2(gm1_list, gm2_list, gds1_list, gds2_list, c1_list, c2_list, cgg_in2_list,
                                            cpar1, cload, res_var, phase_margin, f_unit, seg_gm, seg_diode, seg_ngm)

        sch_info = dict(
            w_dict={'load': load_info['w'], 'in': gm_info['w'], 'tail': tail1_info['w']},
            th_dict={'load': load_info['intent'], 'in': gm_info['intent'], 'tail': tail1_info['intent']},
            stack_dict={'tail': stack_gm, 'in': stack_gm, 'diode': stack_diode, 'ngm': stack_ngm},
            seg_dict={'tail1': seg_gm,
                      'tail2': stage2_results['seg_tail2'],
                      'tailcm': stage2_results['seg_tailcm'],
                      'in': seg_gm,
                      'ref': max(2, seg_gm // max_ref_ratio),
                      'diode1': seg_diode,
                      'ngm1': seg_ngm,
                      'diode2': stage2_results['seg_diode2'],
                      'ngm2': stage2_results['seg_ngm2'],
                      }
        )

        self._amp_info = dict(
            vtail=vtail_list,
            vmid=vd_list,
            vbias=vbias_list,

            vstar=gm_info['vstar'],
            cin=gm_info['cgg'],
            gm1=gm1_list,
            gds1=gds1_list,
            gain1=gain1_list,
            c1=stage2_results['c1'],
            gm2=stage2_results['gm2'],
            gds2=stage2_results['gds2'],
            c2=stage2_results['c2'],

            rz=stage2_results['rz'],
            cf=stage2_results['cf'],
            gain_tot=stage2_results['gain'],
            f_3db=stage2_results['f_3db'],
            f_unity=stage2_results['f_unity'],
            phase_margin=stage2_results['phase_margin'],

            sch_info=sch_info,
        )

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        return self._amp_info

    def design_stage2(self, gm1_list, gm2_list, gds1_list, gds2_list, c1_list, c2_list, cg2_list,
                      cpar1, cload, res_var, phase_margin, f_unit, seg_tail, seg_diode, seg_ngm):
        # step 1: find stage 2 unit size
        seg_gcd = gcd(gcd(seg_tail, seg_diode), seg_ngm)
        if seg_gcd % 2 != 0:
            raise ValueError('All segment numbers must be even.')
        # divide seg_gcd by 2 to make sure all generated segment numbers are even
        seg_gcd //= 2

        # make sure we have enough tail fingers for common mode feedback
        min_size = 2 if seg_tail // seg_gcd == 2 else 1

        bin_iter = BinaryIterator(min_size, None)
        results = {}
        while bin_iter.has_next():
            cur_size = bin_iter.get_next()
            cur_gm2_list, cur_gds2_list, cur_c2_list, cur_cg2_list = self._get_stage2_ss(gm2_list, gds2_list,
                                                                                         c2_list, cg2_list,
                                                                                         cload, seg_gcd, cur_size)

            cur_c1_list = [cpar1 + c1_tmp + cg2_tmp for c1_tmp, cg2_tmp in zip(c1_list, cur_cg2_list)]
            rz, cf, gain_list, f3db_list, funity_list, pm_list = self._find_rz_cf(gm1_list, cur_gm2_list, gds1_list,
                                                                                  cur_gds2_list, cur_c1_list,
                                                                                  cur_c2_list, res_var, phase_margin)

            if min(funity_list) > f_unit:
                seg_tail2_tot = seg_tail // seg_gcd * cur_size
                seg_tail2 = (seg_tail2_tot // 4) * 2
                seg_tailcm = seg_tail2_tot - seg_tail2
                bin_iter.save()
                bin_iter.down()
                results['rz'] = rz
                results['cf'] = cf
                results['gain'] = gain_list
                results['f_3db'] = f3db_list
                results['f_unity'] = funity_list
                results['phase_margin'] = pm_list
                results['gm2'] = cur_gm2_list
                results['gds2'] = cur_gds2_list
                results['c1'] = cur_c1_list
                results['c2'] = cur_c2_list
                results['seg_diode2'] = seg_diode // seg_gcd * cur_size
                results['seg_ngm2'] = seg_ngm // seg_gcd * cur_size
                results['seg_tail2'] = seg_tail2
                results['seg_tailcm'] = seg_tailcm
            else:
                bin_iter.up()

        return results

    @classmethod
    def _get_stage2_ss(cls, gm2_list, gds2_list, c2_list, cg2_list, cload, seg_gcd, cur_size):
        cur_gm2_list, cur_gds2_list, cur_c2_list, cur_cg2_list = [], [], [],  []
        for gm2, gds2, c2, cg2 in zip(gm2_list, gds2_list, c2_list, cg2_list):
            cur_gm2_list.append(gm2 * cur_size / seg_gcd)
            cur_gds2_list.append(gds2 * cur_size / seg_gcd)
            cur_c2_list.append(cload + c2 * cur_size / seg_gcd)
            cur_cg2_list.append(cg2 * cur_size / seg_gcd)

        return cur_gm2_list, cur_gds2_list, cur_c2_list, cur_cg2_list

    def _find_rz_cf(self, gm1_list, gm2_list, gds1_list, gds2_list, c1_list, c2_list, res_var, phase_margin,
                    cap_tol=1e-15, cap_step=10e-15):
        """Find minimum miller cap that stabilizes the system.

        NOTE: This function assume phase of system for any miller cap value will not loop around 360,
        otherwise it may get the phase margin wrong.  This assumption should be valid for this op amp.
        """
        rz_worst = 1 / min(gm2_list)
        rz_nom = rz_worst / (1 - res_var)
        # find maximum Cf needed to stabilize all corners
        cf_min = cap_step
        for gm1, gm2, gds1, gds2, c1, c2 in zip(gm1_list, gm2_list, gds1_list, gds2_list, c1_list, c2_list):
            cir = self._make_circuit(gm1, gm2, gds1, gds2, c1, c2, rz_worst)

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

            # bin_iter is guaranteed to save at least one value, so don't need to worry about cf_min being None
            cf_min = bin_iter.get_last_save()

        # find gain, unity gain bandwidth, and phase margin across corners
        gain_list, f3db_list, funity_list, pm_list = [], [], [], []
        for gm1, gm2, gds1, gds2, c1, c2 in zip(gm1_list, gm2_list, gds1_list, gds2_list, c1_list, c2_list):
            cir = self._make_circuit(gm1, gm2, gds1, gds2, c1, c2, rz_nom)
            cir.add_cap(cf_min, 'vm', 'vo')
            num, den = cir.get_num_den('vi', 'vo')
            pn = np.poly1d(num)
            pd = np.poly1d(den)
            gain_list.append(abs(pn(0) / pd(0)))
            f3db_list.append(get_w_3db(num, den) / 2 / np.pi)
            funity_list.append(get_w_crossings(num, den)[0] / 2 / np.pi)
            pm_list.append(get_stability_margins(num, den)[0])

        return rz_nom, cf_min, gain_list, f3db_list, funity_list, pm_list

    @classmethod
    def _make_circuit(cls, gm1, gm2, gds1, gds2, c1, c2, rz):
        cir = LTICircuit()
        cir.add_conductance(gds1, 'vx', 'gnd')
        cir.add_cap(c1, 'vx', 'gnd')
        cir.add_vccs(gm1, 'vx', 'gnd', 'vi')
        cir.add_conductance(gds2, 'vo', 'gnd')
        cir.add_cap(c2, 'vo', 'gnd')
        cir.add_vccs(gm2, 'vo', 'gnd', 'vx')
        cir.add_res(rz, 'vx', 'vm')
        return cir
