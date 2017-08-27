# -*- coding: utf-8 -*-

"""This module contains design algorithm for a traditional two stage operational amplifier."""

from typing import List, Optional, Dict, Any

import numpy as np
import scipy.optimize as sciopt

from bag.math import gcd
from bag.data.lti import LTICircuit, get_stability_margins, get_w_crossings, get_w_3db
from bag.util.search import FloatBinaryIterator, minimize_cost_golden

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


class StageOneCurrentError(Exception):
    pass


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
            load_db = self._nch_db
            gm_db = self._pch_db
            vds2_list = vout_list
            vb_gm = vdd
            vb_load = 0
        else:
            load_db = self._pch_db
            gm_db = self._nch_db
            vds2_list = [vo - vdd for vo in vout_list]
            vb_gm = 0
            vb_load = vdd

        load = LoadDiodePFB(load_db)
        gm = InputGm(gm_db)
        tail1 = TailStage1(gm_db)

        # design load
        load.design(itarg_list, vds2_list, vstar_load_min, l)
        load_info = load.get_dsn_info()
        vgs_load_list = load_info['vgs']
        gds_load_list = load_info['gds1']
        gm2_list = load_info['gm2']
        stack_diode = load_info['stack_diode']
        stack_ngm = load_info['stack_ngm']
        seg_diode = load_info['seg_diode']
        seg_ngm = load_info['seg_ngm']
        if pmos_input:
            vmid_list = vgs_load_list
        else:
            vmid_list = [vdd - vgs for vgs in vgs_load_list]

        # design input gm
        gm.design(itarg_list, vg_list, vmid_list, gds_load_list, vb_gm, vstar_gm_min, vds_tail_min, l,
                  seg_min=seg_gm_min, stack_list=[stack_ngm])
        gm_info = gm.get_dsn_info()
        gm1_list = gm_info['gm']
        gds_in_list = gm_info['gds']
        vtail_list = gm_info['vs']
        seg_gm = gm_info['seg']
        stack_gm = gm_info['stack']

        gds1_list = [gds_in + gds_load for gds_in, gds_load in zip(gds_in_list, gds_load_list)]
        gain1_list = [gm1 / gds1 for gm1, gds1 in zip(gm1_list, gds1_list)]

        # design stage 1 tail
        tail1.design(itarg_list, vtail_list, vout_list, vb_gm, l, seg_gm, stack_gm)
        tail1_info = tail1.get_dsn_info()
        vbias_list = [vgs_tail + vb_gm for vgs_tail in tail1_info['vgs']]

        # design stage 2 gm
        w_dict = {'load': load_info['w'], 'in': gm_info['w'], 'tail': tail1_info['w']}
        th_dict = {'load': load_info['intent'], 'in': gm_info['intent'], 'tail': tail1_info['intent']}
        stack_dict = {'tail': stack_gm, 'in': stack_gm, 'diode': stack_diode, 'ngm': stack_ngm}
        seg_dict = {'tail1': seg_gm,
                    'in': seg_gm,
                    'ref': max(2, seg_gm // max_ref_ratio),
                    'diode1': seg_diode,
                    'ngm1': seg_ngm,
                    }

        stage2_results = self.design_stage2(gm_db, load_db, vtail_list, vg_list, vmid_list, vout_list, vbias_list,
                                            vb_gm, vb_load, cload, cpar1, w_dict, th_dict, stack_dict, seg_dict,
                                            gm2_list, res_var, phase_margin, f_unit)

        sch_info = dict(
            w_dict=w_dict,
            th_dict=th_dict,
            stack_dict=stack_dict,
            seg_dict=seg_dict,
        )

        self._amp_info = dict(
            vtail=vtail_list,
            vmid=vmid_list,
            vbias=vbias_list,

            vstar=gm_info['vstar'],
            cin=gm_info['cgg'],
            gm1=gm1_list,
            gds1=gds1_list,
            gain1=gain1_list,

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

    def design_stage2(self, gm_db, load_db, vtail_list, vg_list, vmid_list, vout_list, vbias_list,
                      vb_gm, vb_load, cload, cpar1, w_dict, th_dict, stack_dict, seg_dict, gm2_list,
                      res_var, phase_margin, f_unit):

        seg_tail1 = seg_dict['tail1']
        seg_diode1 = seg_dict['diode1']
        seg_ngm1 = seg_dict['ngm1']

        # step 1: find stage 2 unit size
        seg_gcd = gcd(gcd(seg_tail1, seg_diode1), seg_ngm1)
        if seg_gcd % 2 != 0:
            raise ValueError('All segment numbers must be even.')
        # divide seg_gcd by 2 to make sure all generated segment numbers are even
        seg_gcd //= 2

        # make sure we have enough tail fingers for common mode feedback
        min_size = 2 if seg_tail1 // seg_gcd == 2 else 1

        def ac_results_fun(cur_size):
            seg_dict['tail2'] = seg_tail1 // seg_gcd * cur_size
            seg_dict['diode2'] = seg_diode1 // seg_gcd * cur_size
            seg_dict['ngm2'] = seg_ngm1 // seg_gcd * cur_size
            cur_scale2 = cur_size / seg_gcd

            cur_gm2_list = [gm2 * cur_scale2 for gm2 in gm2_list]
            ac_results = self._find_rz_cf(gm_db, load_db, vtail_list, vg_list, vmid_list, vout_list, vbias_list,
                                          vb_gm, vb_load, cload, cpar1, w_dict, th_dict, stack_dict, seg_dict,
                                          cur_scale2, cur_gm2_list, res_var, phase_margin)

            return ac_results

        def funity_fun(cur_size):
            return min(ac_results_fun(cur_size)[0])

        min_result = minimize_cost_golden(funity_fun, f_unit, offset=min_size)

        if min_result.x is None:
            raise StageOneCurrentError('Insufficient stage 1 current.  funity_max = %.4g' % min_result.vmax)

        funity_list, rz_nom, cf_min, gain_list, f3db_list, pm_list = ac_results_fun(min_result.x)

        seg_tail2_tot = seg_dict['tail2']
        seg_tail2 = (seg_tail2_tot // 4) * 2
        seg_tailcm = seg_tail2_tot - seg_tail2
        seg_dict['tail2'] = seg_tail2
        seg_dict['tailcm'] = seg_tailcm
        return dict(
            rz=rz_nom,
            cf=cf_min,
            gain=gain_list,
            f_3db=f3db_list,
            f_unity=funity_list,
            phase_margin=pm_list,
        )

    @classmethod
    def _get_stage2_ss(cls, gm2_list, gds2_list, c2_list, cg2_list, cload, seg_gcd, cur_size):
        cur_gm2_list, cur_gds2_list, cur_c2_list, cur_cg2_list = [], [], [],  []
        for gm2, gds2, c2, cg2 in zip(gm2_list, gds2_list, c2_list, cg2_list):
            cur_gm2_list.append(gm2 * cur_size / seg_gcd)
            cur_gds2_list.append(gds2 * cur_size / seg_gcd)
            cur_c2_list.append(cload + c2 * cur_size / seg_gcd)
            cur_cg2_list.append(cg2 * cur_size / seg_gcd)

        return cur_gm2_list, cur_gds2_list, cur_c2_list, cur_cg2_list

    def _find_rz_cf(self, gm_db, load_db, vtail_list, vg_list, vmid_list, vout_list, vbias_list,
                    vb_gm, vb_load, cload, cpar1, w_dict, th_dict, stack_dict, seg_dict, scale2,
                    gm2_list, res_var, phase_margin, cap_tol=1e-15, cap_step=10e-15, cap_min=1e-15):
        """Find minimum miller cap that stabilizes the system.

        NOTE: This function assume phase of system for any miller cap value will not loop around 360,
        otherwise it may get the phase margin wrong.  This assumption should be valid for this op amp.
        """
        rz_worst = 1 / min(gm2_list)
        rz_nom = rz_worst / (1 - res_var)
        # find maximum Cf needed to stabilize all corners
        cf_min = cap_min
        for env_idx, (vtail, vg, vmid, vout, vbias) in \
                enumerate(zip(vtail_list, vg_list, vmid_list, vout_list, vbias_list)):
            cir = self._make_circuit(env_idx, gm_db, load_db, vtail, vg, vmid, vout, vbias, vb_gm, vb_load,
                                     cload, cpar1, w_dict, th_dict, stack_dict, seg_dict, scale2, rz_worst)

            bin_iter = FloatBinaryIterator(cf_min, None, cap_tol, search_step=cap_step)
            while bin_iter.has_next():
                cur_cf = bin_iter.get_next()
                cir.add_cap(cur_cf, 'outp', 'xp')
                cir.add_cap(cur_cf, 'outn', 'xn')
                num, den = cir.get_num_den('in', 'out')
                cur_pm, _ = get_stability_margins(num, den)
                if cur_pm < phase_margin:
                    bin_iter.up()
                else:
                    bin_iter.save()
                    bin_iter.down()
                cir.add_cap(-cur_cf, 'outp', 'xp')
                cir.add_cap(-cur_cf, 'outn', 'xn')

            # bin_iter is guaranteed to save at least one value, so don't need to worry about cf_min being None
            cf_min = bin_iter.get_last_save()

        # find gain, unity gain bandwidth, and phase margin across corners
        gain_list, f3db_list, funity_list, pm_list = [], [], [], []
        for env_idx, (vtail, vg, vmid, vout, vbias) in \
                enumerate(zip(vtail_list, vg_list, vmid_list, vout_list, vbias_list)):
            cir = self._make_circuit(env_idx, gm_db, load_db, vtail, vg, vmid, vout, vbias, vb_gm, vb_load,
                                     cload, cpar1, w_dict, th_dict, stack_dict, seg_dict, scale2, rz_nom)
            cir.add_cap(cf_min, 'outp', 'xp')
            cir.add_cap(cf_min, 'outn', 'xn')
            num, den = cir.get_num_den('in', 'out')
            pn = np.poly1d(num)
            pd = np.poly1d(den)
            gain_list.append(abs(pn(0) / pd(0)))
            f3db_list.append(get_w_3db(num, den) / 2 / np.pi)
            funity_list.append(get_w_crossings(num, den)[0] / 2 / np.pi)
            pm_list.append(get_stability_margins(num, den)[0])

        return funity_list, rz_nom, cf_min, gain_list, f3db_list, pm_list

    @classmethod
    def _make_circuit(cls, env_idx, gm_db, load_db, vtail, vg, vmid, vout, vbias, vb_gm, vb_load, cload, cpar1,
                      w_dict, th_dict, stack_dict, seg_dict, scale2, rz, neg_cap=False):

        cur_env = gm_db.env_list[env_idx]
        gm_db.set_dsn_params(w=w_dict['tail'], intent=th_dict['tail'], stack=stack_dict['tail'])
        ref_params = gm_db.query(env=cur_env, vbs=0, vds=vbias - vb_gm, vgs=vbias - vb_gm)
        tail1_params = gm_db.query(env=cur_env, vbs=0, vds=vtail - vb_gm, vgs=vbias - vb_gm)
        tail2_params = gm_db.query(env=cur_env, vbs=0, vds=vout - vb_gm, vgs=vbias - vb_gm)
        gm_db.set_dsn_params(w=w_dict['in'], intent=th_dict['in'], stack=stack_dict['in'])
        gm1_params = gm_db.query(env=cur_env, vbs=vb_gm - vtail, vds=vmid - vtail, vgs=vg - vtail)
        load_db.set_dsn_params(w=w_dict['load'], intent=th_dict['load'], stack=stack_dict['diode'])
        diode1_params = load_db.query(env=cur_env, vbs=0, vds=vmid - vb_load, vgs=vmid - vb_load)
        diode2_params = load_db.query(env=cur_env, vbs=0, vds=vout - vb_load, vgs=vmid - vb_load)
        load_db.set_dsn_params(stack=stack_dict['ngm'])
        ngm1_params = load_db.query(env=cur_env, vbs=0, vds=vmid - vb_load, vgs=vmid - vb_load)
        ngm2_params = load_db.query(env=cur_env, vbs=0, vds=vout - vb_load, vgs=vmid - vb_load)

        cir = LTICircuit()
        # stage 1
        cir.add_transistor(ref_params, 'bias', 'bias', 'gnd', 'gnd', fg=seg_dict['ref'], neg_cap=neg_cap)
        cir.add_transistor(tail1_params, 'tail', 'bias', 'gnd', 'gnd', fg=seg_dict['tail1'], neg_cap=neg_cap)
        cir.add_transistor(gm1_params, 'midp', 'inn', 'tail', 'gnd', fg=seg_dict['in'], neg_cap=neg_cap)
        cir.add_transistor(gm1_params, 'midn', 'inp', 'tail', 'gnd', fg=seg_dict['in'], neg_cap=neg_cap)
        cir.add_transistor(diode1_params, 'midp', 'midp', 'gnd', 'gnd', fg=seg_dict['diode1'], neg_cap=neg_cap)
        cir.add_transistor(diode1_params, 'midn', 'midn', 'gnd', 'gnd', fg=seg_dict['diode1'], neg_cap=neg_cap)
        cir.add_transistor(ngm1_params, 'midn', 'midp', 'gnd', 'gnd', fg=seg_dict['ngm1'], neg_cap=neg_cap)
        cir.add_transistor(ngm1_params, 'midp', 'midn', 'gnd', 'gnd', fg=seg_dict['ngm1'], neg_cap=neg_cap)

        # stage 2
        cir.add_transistor(tail2_params, 'outp', 'bias', 'gnd', 'gnd', fg=seg_dict['tail2'] * scale2, neg_cap=neg_cap)
        cir.add_transistor(tail2_params, 'outn', 'bias', 'gnd', 'gnd', fg=seg_dict['tail2'] * scale2, neg_cap=neg_cap)
        cir.add_transistor(diode2_params, 'outp', 'midn', 'gnd', 'gnd', fg=seg_dict['diode2'] * scale2, neg_cap=neg_cap)
        cir.add_transistor(diode2_params, 'outn', 'midp', 'gnd', 'gnd', fg=seg_dict['diode2'] * scale2, neg_cap=neg_cap)
        cir.add_transistor(ngm2_params, 'outp', 'midn', 'gnd', 'gnd', fg=seg_dict['ngm2'] * scale2, neg_cap=neg_cap)
        cir.add_transistor(ngm2_params, 'outn', 'midp', 'gnd', 'gnd', fg=seg_dict['ngm2'] * scale2, neg_cap=neg_cap)

        # parasitic cap
        cir.add_cap(cpar1, 'midp', 'gnd')
        cir.add_cap(cpar1, 'midn', 'gnd')
        # load cap
        cir.add_cap(cload, 'outp', 'gnd')
        cir.add_cap(cload, 'outn', 'gnd')
        # feedback resistors
        cir.add_res(rz, 'xp', 'midn')
        cir.add_res(rz, 'xn', 'midp')
        # diff-to-single conversion
        cir.add_vcvs(0.5, 'inp', 'gnd', 'in', 'gnd')
        cir.add_vcvs(-0.5, 'inn', 'gnd', 'in', 'gnd')
        cir.add_vcvs(1, 'out', 'gnd', 'outp', 'outn')

        return cir
