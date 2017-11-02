# -*- coding: utf-8 -*-

"""This module contains design algorithm for a traditional two stage operational amplifier."""

from typing import List, Optional, Dict, Any, Tuple, Sequence

from copy import deepcopy

import numpy as np
import scipy.optimize as sciopt
import scipy.interpolate as interp

from bag.core import BagProject, Testbench
from bag.math import gcd
from bag.data.core import Waveform
from bag.data.lti import LTICircuit, get_stability_margins, get_w_crossings, get_w_3db
from bag.util.search import FloatBinaryIterator, BinaryIterator, minimize_cost_golden
from bag.simulation.core import SimulationManager

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
               i1_unit,  # type: List[float]
               i1_min_size,  # type: int
               vg_list,  # type: List[float]
               vout_list,  # type: List[float]
               cpar1,  # type: float
               cload,  # type: float
               f_unit,  # type: float
               phase_margin,  # type: float
               res_var,  # type: float
               l,  # type: float
               vstar_gm_min,  # type: float
               ft_load_scale,  # type: float
               vds_tail_min,  # type: float
               seg_gm_min,  # type: int
               vdd,  # type: float
               pmos_input=True,  # type: bool
               max_ref_ratio=20,  # type: int
               load_stack_list=None,  # type: Optional[List[int]]
               ):
        # type: (...) -> None

        # binary search for minimum stage 1 current,
        i1_size_iter = BinaryIterator(i1_min_size, None)
        i1_size_opt, opt_info = None, None
        while i1_size_iter.has_next():
            i1_size = i1_size_iter.get_next()
            print('trying i1_size = %d' % i1_size)
            try:
                self._design_with_itarg(i1_size, i1_unit, vg_list, vout_list, cpar1, cload,
                                        f_unit, phase_margin, res_var, l, vstar_gm_min,
                                        ft_load_scale, vds_tail_min, seg_gm_min,
                                        vdd, pmos_input, max_ref_ratio, load_stack_list)
                success = True
            except StageOneCurrentError as err:
                print(err)
                success = False

            if success:
                print('success')
                opt_info = self._amp_info
                i1_size_opt = i1_size
                i1_size_iter.down()
            else:
                i1_size_iter.up()

        # linear search to find optimal scale2
        scale2_int_max = int(opt_info['scale2'])
        if scale2_int_max == opt_info['scale2']:
            scale2_int_max -= 1

        last_i1_size = i1_size_opt
        print('i1_size = %d, scale2 = %.4g' % (i1_size_opt, opt_info['scale2']))
        for scale2_test in range(scale2_int_max, 0, -1):
            i1_size_test = int(np.floor(i1_size_opt * (1 + opt_info['scale2']) / (1 + scale2_test)))
            if i1_size_test <= last_i1_size or scale2_test == opt_info['scale2']:
                continue
            print('testing i1_size = %d, scale2 = %.4g' % (i1_size_test, scale2_test))
            try:
                self._design_with_itarg(i1_size_test, i1_unit, vg_list, vout_list, cpar1, cload,
                                        f_unit, phase_margin, res_var, l, vstar_gm_min,
                                        ft_load_scale, vds_tail_min, seg_gm_min,
                                        vdd, pmos_input, max_ref_ratio, load_stack_list)
            except StageOneCurrentError as err:
                print(err)
                continue
            if self._amp_info['scale2'] <= scale2_test:
                # found new minimum.  close in to find optimal i1 size
                opt_info = self._amp_info
                i1_size_opt = i1_size_test
                print('update: i1_size = %d, scale2 = %.4g' % (i1_size_opt, opt_info['scale2']))
                i1_size_iter = BinaryIterator(last_i1_size + 1, i1_size_test)
                while i1_size_iter.has_next():
                    i1_size_cur_opt = i1_size_iter.get_next()
                    print('testing i1_size = %d' % i1_size_cur_opt)
                    try:
                        self._design_with_itarg(i1_size_cur_opt, i1_unit, vg_list, vout_list, cpar1, cload,
                                                f_unit, phase_margin, res_var, l, vstar_gm_min,
                                                ft_load_scale, vds_tail_min, seg_gm_min,
                                                vdd, pmos_input, max_ref_ratio, load_stack_list)

                        if self._amp_info['scale2'] <= opt_info['scale2']:
                            opt_info = self._amp_info
                            i1_size_opt = i1_size_cur_opt
                            print('update: i1_size = %d, scale2 = %.4g' % (i1_size_opt, opt_info['scale2']))
                            i1_size_iter.down()
                        else:
                            i1_size_iter.up()

                    except StageOneCurrentError as err:
                        print(err)
                        i1_size_iter.up()

            last_i1_size = i1_size_test

        self._amp_info = opt_info

    def _design_with_itarg(self,
                           i1_size,  # type: int
                           i1_unit,  # type: List[float]
                           vg_list,  # type: List[float]
                           vout_list,  # type: List[float]
                           cpar1,  # type: float
                           cload,  # type: float
                           f_unit,  # type: float
                           phase_margin,  # type: float
                           res_var,  # type: float
                           l,  # type: float
                           vstar_gm_min,  # type: float
                           ft_load_scale,  # type: float
                           vds_tail_min,  # type: float
                           seg_gm_min,  # type: int
                           vdd,  # type: float
                           pmos_input,  # type: bool
                           max_ref_ratio,  # type: int
                           load_stack_list,  # type: Optional[List[int]]
                           ):
        # type: (...) -> None
        itarg_list = [i1 * i1_size for i1 in i1_unit]

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
        print('designing load')
        load.design(itarg_list, vds2_list, ft_load_scale * f_unit, l, stack_list=load_stack_list)
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
        print('designing input gm')
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
        print('designing tail')
        tail1.design(itarg_list, vtail_list, vout_list, vb_gm, l, seg_gm, stack_gm)
        tail1_info = tail1.get_dsn_info()
        vbias_list = [vgs_tail + vb_gm for vgs_tail in tail1_info['vgs']]

        # design stage 2 gm
        w_dict = {'load': load_info['w'], 'in': gm_info['w'], 'tail': tail1_info['w']}
        th_dict = {'load': load_info['intent'], 'in': gm_info['intent'], 'tail': tail1_info['intent']}
        stack_dict = {'tail': stack_gm, 'in': stack_gm, 'diode': stack_diode, 'ngm': stack_ngm}
        seg_dict = {'tail1': seg_gm,
                    'in': seg_gm,
                    'diode1': seg_diode,
                    'ngm1': seg_ngm,
                    }

        print('designing stage 2')
        stage2_results = self._design_stage2(gm_db, load_db, vtail_list, vg_list, vmid_list, vout_list, vbias_list,
                                             vb_gm, vb_load, cload, cpar1, w_dict, th_dict, stack_dict, seg_dict,
                                             gm2_list, res_var, phase_margin, f_unit, max_ref_ratio)

        scale2 = seg_dict['diode2'] / seg_dict['diode1']
        scaler = seg_dict['ref'] / seg_dict['tail1']
        itot_list = [(2 * (1 + scale2) + scaler) * itarg for itarg in itarg_list]

        layout_info = dict(
            w_dict=w_dict,
            th_dict=th_dict,
            stack_dict=stack_dict,
            seg_dict=seg_dict,
        )

        self._amp_info = dict(
            i1_size=i1_size,
            scale2=scale2,
            scaler=scaler,
            vtail=vtail_list,
            vmid=vmid_list,
            vbias=vbias_list,
            itot=itot_list,

            vstar=gm_info['vstar'],
            cin=gm_info['cgg'],
            gm1=gm1_list,
            gds1=gds1_list,
            gain1=gain1_list,

            rfb=stage2_results['rz'],
            cfb=stage2_results['cf'],
            gain_tot=stage2_results['gain'],
            f_3db=stage2_results['f_3db'],
            f_unit=stage2_results['f_unity'],
            phase_margin=stage2_results['phase_margin'],

            layout_info=layout_info,
        )

        print('done')

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        return self._amp_info

    def get_specs_verification(self, top_specs):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        top_specs = deepcopy(top_specs)
        dsn_specs = top_specs['dsn_specs']

        ibias = dsn_specs['i1_unit'][0] * self._amp_info['i1_size'] * self._amp_info['scaler']
        vdd = dsn_specs['vdd']
        vindc = dsn_specs['vg_list'][0]
        voutdc = dsn_specs['vout_list'][0]
        f_unit = dsn_specs['f_unit']
        gain_max = max(self._amp_info['gain_tot'])
        f_bw_log = int(np.floor(np.log10(f_unit / gain_max)))
        f_unit_log = int(np.ceil(np.log10(f_unit)))

        top_specs['layout_params'].update(self._amp_info['layout_info'])

        top_specs['wrapper_params']['cload'] = dsn_specs['cload']
        top_specs['wrapper_params']['vdd'] = vdd

        top_specs['feedback_params']['cfb'] = self._amp_info['cfb']
        top_specs['feedback_params']['rfb'] = self._amp_info['rfb']

        top_specs['tb_dc']['tb_params']['vimax'] = vdd
        top_specs['tb_dc']['tb_params']['vimin'] = -vdd
        top_specs['tb_dc']['tb_params']['vindc'] = vindc
        top_specs['tb_dc']['tb_params']['voutcm'] = voutdc
        top_specs['tb_dc']['tb_params']['ibias'] = ibias
        top_specs['tb_dc']['tb_params']['vdd'] = vdd
        top_specs['tb_dc']['tb_params']['voutref'] = voutdc
        top_specs['tb_dc']['tb_params']['vout_start'] = -vdd + 0.15
        top_specs['tb_dc']['tb_params']['vout_stop'] = vdd - 0.15

        top_specs['tb_ac']['tb_params']['vincm'] = vindc
        top_specs['tb_ac']['tb_params']['voutcm'] = voutdc
        top_specs['tb_ac']['tb_params']['ibias'] = ibias
        top_specs['tb_ac']['tb_params']['vdd'] = vdd
        top_specs['tb_ac']['tb_params']['vinac'] = 1.0
        top_specs['tb_ac']['tb_params']['vindc'] = 0.0
        top_specs['tb_ac']['tb_params']['fstart'] = 10 ** f_bw_log
        top_specs['tb_ac']['tb_params']['fstop'] = 10 ** (f_unit_log + 1)

        return top_specs

    def _design_stage2(self, gm_db, load_db, vtail_list, vg_list, vmid_list, vout_list, vbias_list,
                       vb_gm, vb_load, cload, cpar1, w_dict, th_dict, stack_dict, seg_dict, gm2_list,
                       res_var, phase_margin, f_unit, max_ref_ratio):

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
                                          cur_gm2_list, res_var, phase_margin)

            return ac_results

        def funity_fun(cur_size):
            ac_results_tmp = ac_results_fun(cur_size)
            fu_list = ac_results_tmp[0]
            if fu_list is None:
                return -1
            # noinspection PyTypeChecker
            ans = min(fu_list)
            return ans

        # find min_size such that amplifier is stable
        min_bin_iter = BinaryIterator(min_size, None)
        while min_bin_iter.has_next():
            test_size = min_bin_iter.get_next()
            test_fu = funity_fun(test_size)
            if test_fu >= 0:
                min_bin_iter.save()
                min_bin_iter.down()
            else:
                min_bin_iter.up()

        min_result = minimize_cost_golden(funity_fun, f_unit, offset=min_bin_iter.get_last_save())

        if min_result.x is None:
            msg = 'Insufficient stage 1 current.  funity_max=%.4g'
            raise StageOneCurrentError(msg % min_result.vmax)

        funity_list, rz_nom, cf_min, gain_list, f3db_list, pm_list = ac_results_fun(min_result.x)

        seg_tail2_tot = seg_dict['tail2']
        seg_tail2 = (seg_tail2_tot // 4) * 2
        seg_tailcm = seg_tail2_tot - seg_tail2
        seg_tail_tot = 2 * (seg_dict['tail1'] + seg_tail2)
        seg_dict['tail2'] = seg_tail2
        seg_dict['tailcm'] = seg_tailcm
        seg_dict['ref'] = max(2, -((-seg_tail_tot // max_ref_ratio) // 2) * 2)
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
        cur_gm2_list, cur_gds2_list, cur_c2_list, cur_cg2_list = [], [], [], []
        for gm2, gds2, c2, cg2 in zip(gm2_list, gds2_list, c2_list, cg2_list):
            cur_gm2_list.append(gm2 * cur_size / seg_gcd)
            cur_gds2_list.append(gds2 * cur_size / seg_gcd)
            cur_c2_list.append(cload + c2 * cur_size / seg_gcd)
            cur_cg2_list.append(cg2 * cur_size / seg_gcd)

        return cur_gm2_list, cur_gds2_list, cur_c2_list, cur_cg2_list

    def _find_rz_cf(self, gm_db, load_db, vtail_list, vg_list, vmid_list, vout_list, vbias_list,
                    vb_gm, vb_load, cload, cpar1, w_dict, th_dict, stack_dict, seg_dict,
                    gm2_list, res_var, phase_margin, cap_tol=1e-15, cap_step=10e-15, cap_min=1e-15, cap_max=1e-9):
        """Find minimum miller cap that stabilizes the system.

        NOTE: This function assume phase of system for any miller cap value will not loop around 360,
        otherwise it may get the phase margin wrong.  This assumption should be valid for this op amp.
        """
        gz_worst = float(min(gm2_list))
        gz_nom = gz_worst * (1 - res_var)
        # find maximum Cf needed to stabilize all corners
        cf_min = cap_min
        for env_idx, (vtail, vg, vmid, vout, vbias) in \
                enumerate(zip(vtail_list, vg_list, vmid_list, vout_list, vbias_list)):
            cir = self._make_circuit(env_idx, gm_db, load_db, vtail, vg, vmid, vout, vbias, vb_gm, vb_load,
                                     cload, cpar1, w_dict, th_dict, stack_dict, seg_dict, gz_worst)

            bin_iter = FloatBinaryIterator(cf_min, None, cap_tol, search_step=cap_step)
            while bin_iter.has_next():
                cur_cf = bin_iter.get_next()
                cir.add_cap(cur_cf, 'outp', 'xp')
                cir.add_cap(cur_cf, 'outn', 'xn')
                num, den = cir.get_num_den('in', 'out')
                cur_pm, _ = get_stability_margins(num, den)
                if cur_pm < phase_margin:
                    if cur_cf > cap_max:
                        # no way to make amplifier stable, just return
                        return None, None, None, None, None, None
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
                                     cload, cpar1, w_dict, th_dict, stack_dict, seg_dict, gz_nom)
            cir.add_cap(cf_min, 'outp', 'xp')
            cir.add_cap(cf_min, 'outn', 'xn')
            num, den = cir.get_num_den('in', 'out')
            pn = np.poly1d(num)
            pd = np.poly1d(den)
            gain_list.append(abs(pn(0) / pd(0)))
            f3db_list.append(get_w_3db(num, den) / 2 / np.pi)
            funity_list.append(get_w_crossings(num, den)[0] / 2 / np.pi)
            pm_list.append(get_stability_margins(num, den)[0])

        return funity_list, 1 / gz_nom, cf_min, gain_list, f3db_list, pm_list

    @classmethod
    def _make_circuit(cls, env_idx, gm_db, load_db, vtail, vg, vmid, vout, vbias, vb_gm, vb_load, cload, cpar1,
                      w_dict, th_dict, stack_dict, seg_dict, gz, neg_cap=False, no_fb=False):

        cur_env = gm_db.env_list[env_idx]
        gm_db.set_dsn_params(w=w_dict['tail'], intent=th_dict['tail'], stack=stack_dict['tail'])
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
        cir.add_transistor(tail1_params, 'tail', 'gnd', 'gnd', 'gnd', fg=seg_dict['tail1'], neg_cap=neg_cap)
        cir.add_transistor(gm1_params, 'midp', 'inn', 'tail', 'gnd', fg=seg_dict['in'], neg_cap=neg_cap)
        cir.add_transistor(gm1_params, 'midn', 'inp', 'tail', 'gnd', fg=seg_dict['in'], neg_cap=neg_cap)
        cir.add_transistor(diode1_params, 'midp', 'midp', 'gnd', 'gnd', fg=seg_dict['diode1'], neg_cap=neg_cap)
        cir.add_transistor(diode1_params, 'midn', 'midn', 'gnd', 'gnd', fg=seg_dict['diode1'], neg_cap=neg_cap)
        cir.add_transistor(ngm1_params, 'midn', 'midp', 'gnd', 'gnd', fg=seg_dict['ngm1'], neg_cap=neg_cap)
        cir.add_transistor(ngm1_params, 'midp', 'midn', 'gnd', 'gnd', fg=seg_dict['ngm1'], neg_cap=neg_cap)

        # stage 2
        cir.add_transistor(tail2_params, 'outp', 'gnd', 'gnd', 'gnd', fg=seg_dict['tail2'], neg_cap=neg_cap)
        cir.add_transistor(tail2_params, 'outn', 'gnd', 'gnd', 'gnd', fg=seg_dict['tail2'], neg_cap=neg_cap)
        cir.add_transistor(diode2_params, 'outp', 'midn', 'gnd', 'gnd', fg=seg_dict['diode2'], neg_cap=neg_cap)
        cir.add_transistor(diode2_params, 'outn', 'midp', 'gnd', 'gnd', fg=seg_dict['diode2'], neg_cap=neg_cap)
        cir.add_transistor(ngm2_params, 'outp', 'midn', 'gnd', 'gnd', fg=seg_dict['ngm2'], neg_cap=neg_cap)
        cir.add_transistor(ngm2_params, 'outn', 'midp', 'gnd', 'gnd', fg=seg_dict['ngm2'], neg_cap=neg_cap)

        # parasitic cap
        cir.add_cap(cpar1, 'midp', 'gnd')
        cir.add_cap(cpar1, 'midn', 'gnd')
        # load cap
        cir.add_cap(cload, 'outp', 'gnd')
        cir.add_cap(cload, 'outn', 'gnd')
        # feedback resistors
        if not no_fb:
            cir.add_conductance(gz, 'xp', 'midn')
            cir.add_conductance(gz, 'xn', 'midp')
        # diff-to-single conversion
        cir.add_vcvs(0.5, 'inp', 'gnd', 'in', 'gnd')
        cir.add_vcvs(-0.5, 'inn', 'gnd', 'in', 'gnd')
        cir.add_vcvs(1, 'out', 'gnd', 'outp', 'outn')

        return cir


class OpAmpTwoStageChar(SimulationManager):
    def __init__(self, prj, spec_file):
        # type: (Optional[BagProject], str) -> None
        super(OpAmpTwoStageChar, self).__init__(prj, spec_file)
        combo_list = list(self.get_combinations_iter())

        if len(combo_list) != 1:
            raise ValueError('This characterization class does not support sweeping design parameters.')

        self._val_list = combo_list[0]

    def configure_tb(self, tb_type, tb, val_list):
        # type: (str, Testbench, Tuple[Any, ...]) -> None
        tb_specs = self.specs[tb_type]
        sim_envs = self.specs['sim_envs']
        view_name = self.specs['view_name']
        impl_lib = self.specs['impl_lib']
        dsn_name_base = self.specs['dsn_name_base']
        tb_params_real = self.specs['feedback_params'].copy()

        tb_params = tb_specs['tb_params']
        dsn_name = self.get_instance_name(dsn_name_base, val_list)

        tb.set_simulation_environments(sim_envs)
        tb.set_simulation_view(impl_lib, dsn_name, view_name)

        tb_params_real.update(tb_params)
        for key, val in tb_params_real.items():
            if isinstance(val, list):
                tb.set_sweep_parameter(key, values=val)
            else:
                tb.set_parameter(key, val)

    def find_cfb(self, min_scale=1.0, max_scale=2.0, num_pts=11, extract=True, gen_dsn=True):
        tb_type = 'tb_ac'
        feedback_params = self.specs['feedback_params']
        tb_ac_specs = self.specs[tb_type]
        dsn_specs = self.specs['dsn_specs']

        res_var = dsn_specs['res_var']
        phase_margin = dsn_specs['phase_margin']

        rfb0 = feedback_params['rfb']
        cfb0 = feedback_params['cfb']
        # noinspection PyUnresolvedReferences
        cfb_list = np.linspace(cfb0 * min_scale, cfb0 * max_scale, num_pts).tolist()
        tb_ac_specs['tb_params']['cfb'] = cfb_list
        tb_ac_specs['tb_params']['rfb'] = rfb0 * (1 - res_var)

        if gen_dsn:
            self.create_designs(tb_type=tb_type, extract=extract)
        else:
            self.run_simulations(tb_type)

        # determine Cfb that stabilizes amplifier
        cfb = self.find_min_cfb(phase_margin)

        # get actual performace
        tb_ac_specs['tb_params']['cfb'] = cfb
        tb_ac_specs['tb_params']['rfb'] = rfb0

        self.run_simulations(tb_type)
        corner_list, f_unity_list, pm_list = self.process_ac_data()

        return cfb, corner_list, f_unity_list, pm_list

    def process_dc_data(self, plot=False):
        tb_type = 'tb_dc'
        axis_names = ['corner', 'voutref']

        results = self.get_sim_results(tb_type, self._val_list)
        corner_list = results['corner']
        corner_sort_arg = np.argsort(corner_list)  # type: Sequence[int]
        corner_list = corner_list[corner_sort_arg].tolist()

        sweep_vars = results['sweep_params']['vout']
        order = [sweep_vars.index(name) for name in axis_names]
        vout_data = np.transpose(results['vout'], axes=order)
        vin_data = np.transpose(results['vin'], axes=order)

        gain_list, vout_vec_list, gain_vec_list, vin_vec_list = [], [], [], []
        for corner_idx in corner_sort_arg:
            vout_vec = vout_data[corner_idx, :]
            vin_vec = vin_data[corner_idx, :]
            cur_vin, vin_arg = np.unique(vin_vec, return_index=True)
            cur_vout = vout_vec[vin_arg]

            vout_fun = interp.InterpolatedUnivariateSpline(cur_vin, cur_vout)
            vout_diff_fun = vout_fun.derivative(1)
            gain_list.append(vout_diff_fun([0]))
            vin_vec_list.append(cur_vin)
            vout_vec_list.append(cur_vout)
            gain_vec_list.append(vout_diff_fun(cur_vin))

        gain_str = ', '.join(('%.4g' % gain for gain in gain_list))
        print('gain=[%s]' % gain_str)

        if plot:
            import matplotlib.pyplot as plt
            plt.ticklabel_format(style='sci', axis='both', scilimits=(-1, 2))
            f, (ax1, ax2) = plt.subplots(2, sharex='all')
            ax1.set_title('Vout v.s. Vin')
            ax1.set_ylabel('Vout (V)')
            ax2.set_title('Gain v.s. Vin')
            ax2.set_xlabel('Vin (V)')
            for corner, vout_vec, gain_vec, vin_vec in \
                    zip(corner_list, vout_vec_list, gain_vec_list, vin_vec_list):
                ax1.plot(vin_vec, vout_vec, label=corner)
                ax2.plot(vin_vec, gain_vec, label=corner)

            ax1.legend()
            ax2.legend()

        return corner_list, gain_list

    def process_ac_data(self, plot=False):
        tb_type = 'tb_ac'
        axis_names = ['corner', 'freq']

        results = self.get_sim_results(tb_type, self._val_list)
        corner_list = results['corner']
        corner_sort_arg = np.argsort(corner_list)  # type: Sequence[int]
        corner_list = corner_list[corner_sort_arg].tolist()

        # rearrange array axis
        sweep_vars = results['sweep_params']['vout_ac']
        order = [sweep_vars.index(name) for name in axis_names]
        vout_data = np.transpose(results['vout_ac'], axes=order)

        # determine minimum cfb
        freq_vec = results['freq']
        f_unity_list, pm_list, vout_vec_list = [], [], []
        for corner_idx in corner_sort_arg:
            vout_vec = vout_data[corner_idx, :]
            f_unity, pm = self._get_ac_info(freq_vec, vout_vec)
            vout_vec_list.append(vout_vec)
            f_unity_list.append(f_unity)
            pm_list.append(pm)

        f_unity_str = ', '.join(('%.4g' % f_unity for f_unity in f_unity_list))
        pm_str = ', '.join(('%.4g' % pm for pm in pm_list))
        print('f_unity=[%s]' % f_unity_str)
        print('phase_margin=[%s]' % pm_str)

        if plot:
            import matplotlib.pyplot as plt
            plt.ticklabel_format(style='sci', axis='both', scilimits=(-1, 2))
            f, (ax1, ax2) = plt.subplots(2, sharex='all')
            ax1.set_title('Magnitude v.s. Frequency')
            ax1.set_ylabel('Magnitude (dB)')
            ax2.set_title('Phase v.s. Frequency')
            ax2.set_xlabel('Phase (Degrees)')
            for corner, vout_vec in zip(corner_list, vout_vec_list):
                mag_vec = 20 * np.log10(np.abs(vout_vec))
                ang_vec = np.rad2deg(np.unwrap(np.angle(vout_vec)))
                ax1.semilogx(freq_vec, mag_vec, label=corner)
                ax2.semilogx(freq_vec, ang_vec, label=corner)

            ax1.legend()
            ax2.legend()

        return corner_list, f_unity_list, pm_list

    def find_min_cfb(self, phase_margin):
        tb_type = 'tb_ac'
        axis_names = ['corner', 'cfb', 'freq']
        dsn_name_base = self.specs['dsn_name_base']

        dsn_name = self.get_instance_name(dsn_name_base, self._val_list)

        results = self.get_sim_results(tb_type, self._val_list)
        corner_list = results['corner']
        corner_sort_arg = np.argsort(corner_list)  # type: Sequence[int]

        # rearrange array axis
        sweep_vars = results['sweep_params']['vout_ac']
        order = [sweep_vars.index(name) for name in axis_names]
        vout_data = np.transpose(results['vout_ac'], axes=order)

        # determine minimum cfb
        cfb_vec = results['cfb']
        freq_vec = results['freq']
        cfb_idx_min = 0
        for corner_idx in corner_sort_arg:
            bin_iter = BinaryIterator(cfb_idx_min, cfb_vec.size)
            while bin_iter.has_next():
                cur_cfb_idx = bin_iter.get_next()
                vout_vec = vout_data[corner_idx, cur_cfb_idx, :]
                _, pm = self._get_ac_info(freq_vec, vout_vec)
                if pm >= phase_margin:
                    bin_iter.save()
                    bin_iter.down()
                else:
                    bin_iter.up()
            cfb_idx_min = bin_iter.get_last_save()
            if cfb_idx_min is None:
                # No solution; cannot make amplifier stable
                break

        if cfb_idx_min is None:
            cfb = None
            print('dsn = %s, cfb = None' % dsn_name)
        else:
            cfb = cfb_vec[cfb_idx_min]
            print('dsn = %s, cfb = %.4g' % (dsn_name, cfb))

        return cfb

    @classmethod
    def _get_ac_info(cls, freq_vec, vout_vec, lf_tol=1e-6):
        xvec = np.log10(freq_vec)
        mag_vec = np.log10(np.abs(vout_vec))
        phase_vec = np.rad2deg(np.unwrap(np.angle(vout_vec)))
        mag_wv = Waveform(xvec, mag_vec, lf_tol)
        phase_wv = Waveform(xvec, phase_vec, lf_tol)

        lf_unity = mag_wv.get_crossing(0)
        phase0 = phase_wv(xvec[0])
        phase = phase_wv(lf_unity)

        return 10.0**lf_unity, 180 - (phase0 - phase)
