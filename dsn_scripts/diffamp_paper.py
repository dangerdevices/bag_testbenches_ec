# -*- coding: utf-8 -*-

"""This script designs a simple diff amp with gain/bandwidth spec for BAG CICC paper."""

import math

import numpy as np
import scipy.optimize as sciopt

from bag.io import read_yaml
from bag.util.search import BinaryIterator

from ckt_dsn_ec.mos.core import MOSDBDiscrete


def design_amp(amp_specs, nch_db, pch_db):
    sim_env = amp_specs['sim_env']
    vdd = amp_specs['vdd']
    vtail = amp_specs['vtail']
    vgs_res = amp_specs['vgs_res']
    gain_min = amp_specs['gain_min']
    gain_margin = amp_specs['gain_margin']
    bw_min = amp_specs['bw_min']
    cload = amp_specs['cload']

    fun_ibiasn = nch_db.get_function('ibias', env=sim_env)
    fun_gmn = nch_db.get_function('gm', env=sim_env)
    fun_gdsn = nch_db.get_function('gds', env=sim_env)
    fun_cdn = nch_db.get_function('cdb', env=sim_env) + nch_db.get_function('cds', env=sim_env)
    fun_cgsn = nch_db.get_function('cgs', env=sim_env)

    fun_ibiasp = pch_db.get_function('ibias', env=sim_env)
    fun_gdsp = pch_db.get_function('gds', env=sim_env)
    fun_cdp = pch_db.get_function('cdd', env=sim_env)

    vgsn_idx = nch_db.get_fun_arg_index('vgs')
    vgsn_min, vgsn_max = fun_ibiasn.get_input_range(vgsn_idx)
    num_pts = int(math.ceil((vgsn_max - vgsn_min) / vgs_res))
    vgs_list = np.linspace(vgsn_min, vgsn_max, num_pts + 1).tolist()

    vgsp_idx = pch_db.get_fun_arg_index('vgs')
    vgsp_min, vgsp_max = fun_ibiasp.get_input_range(vgsp_idx)

    # sweep vgs, find best point
    performance = None
    for vgsn_cur in vgs_list:
        vout = vgsn_cur + vtail

        seg_in_iter = BinaryIterator(2, None, step=2)
        narg = nch_db.get_fun_arg(vgs=vgsn_cur, vds=vgsn_cur, vbs=vtail)
        ibiasn_unit = fun_ibiasn(narg)
        gmn_unit = fun_gmn(narg)
        gdsn_unit = fun_gdsn(narg)
        cdn_unit = fun_cdn(narg)
        cgsn_unit = fun_cgsn(narg)

        # check there's gain solution
        parg = pch_db.get_fun_arg(vgs=vgsp_min, vds=vout - vdd, vbs=0)
        ibiasp_unit_test = fun_ibiasp(parg)
        gdsp_unit_test = fun_gdsp(parg)
        gain_max = gmn_unit / ibiasn_unit / (gdsn_unit / ibiasn_unit + gdsp_unit_test / ibiasp_unit_test)
        if gain_max < gain_min + gain_margin:
            continue

        # sweep gm size
        while seg_in_iter.has_next():
            seg_in = seg_in_iter.get_next()
            ibiasn = seg_in * ibiasn_unit
            gmn = seg_in * gmn_unit
            gdsn = seg_in * gdsn_unit

            # sweep load size
            seg_load_iter = BinaryIterator(2, None, step=2)
            while seg_load_iter.has_next():
                seg_load = seg_load_iter.get_next()
                vbp = find_load_bias(pch_db, vdd, vout, vgsp_min, vgsp_max, ibiasn, seg_load, fun_ibiasp)
                if vbp is None:
                    seg_load_iter.up()
                else:
                    parg = pch_db.get_fun_arg(vgs=vbp - vdd, vds=vout - vdd, vbs=0)
                    gdsp = seg_load * fun_gdsp(parg)
                    if gmn / (gdsp + gdsn) >= gain_min:
                        seg_load_iter.save_info((vbp, parg))
                        seg_load_iter.down()
                    else:
                        seg_load_iter.up()

            seg_load = seg_load_iter.get_last_save()
            vbp, parg = seg_load_iter.get_last_save_info()
            gdsp = seg_load * fun_gdsp(parg)
            cdp = seg_load * fun_cdp(parg)

            cdn = seg_in * cdn_unit
            cgsn = seg_in * cgsn_unit

            ro_cur = 1 / (gdsp + gdsn)
            gain_cur = gmn * ro_cur
            cpar_cur = cdn + cdp + (1 + 1 / gain_cur) * cgsn

            # check intrinsic bandwidth good
            if 1 / (ro_cur * cpar_cur * 2 * np.pi) < bw_min:
                break

            cload_cur = cload + cpar_cur
            bw_cur = 1 / (ro_cur * cload_cur * 2 * np.pi)
            if bw_cur < bw_min:
                seg_in_iter.up()
            else:
                seg_in_iter.save_info((seg_load, vbp, ibiasn, gain_cur, bw_cur))
                seg_in_iter.down()

        if seg_in_iter.get_last_save() is None:
            continue

        seg_in = seg_in_iter.get_last_save()
        seg_load, vbp, ibiasn, gain_cur, bw_cur = seg_in_iter.get_last_save_info()
        if performance is None or performance[0] > ibiasn:
            performance = (ibiasn, gain_cur, bw_cur, seg_in, seg_load, vgsn_cur, vbp)

    if performance is None:
        return None
    ibias_opt, gain_opt, bw_opt, seg_in, seg_load, vgs_in, vload = performance
    vio = vtail + vgs_in
    vbias = find_tail_bias(fun_ibiasn, nch_db, vtail, vgsn_min, vgsn_max, seg_in, ibias_opt)

    return ibias_opt, gain_opt, bw_opt, seg_in, seg_load, vbias, vio, vload


def find_tail_bias(fun_ibiasn, nch_db, vtail, vgs_min, vgs_max, seg_tail, itarg):
    def fun_zero(vgs):
        narg = nch_db.get_fun_arg(vgs=vgs, vds=vtail, vbs=0)
        return fun_ibiasn(narg) * seg_tail - itarg

    vbias = sciopt.brentq(fun_zero, vgs_min, vgs_max)  # type: float
    return vbias


def find_load_bias(pch_db, vdd, vout, vgsp_min, vgsp_max, itarg, seg_load, fun_ibiasp):
    def fun_zero(vbias):
        parg = pch_db.get_fun_arg(vgs=vbias - vdd, vds=vout - vdd, vbs=0)
        return fun_ibiasp(parg) * seg_load - itarg

    vbias_min = vdd + vgsp_max
    vbias_max = vdd + vgsp_min

    try:
        vbias_opt = sciopt.brentq(fun_zero, vbias_min, vbias_max)  # type: float
        return vbias_opt
    except ValueError:
        return None


def run_main():
    nch_config = 'mos_char_specs/nch_w4_amp.yaml'
    pch_config = 'mos_char_specs/pch_w4_amp.yaml'
    amp_specs = 'specs_dsn/diffamp_paper.yaml'

    amp_specs = read_yaml(amp_specs)

    print('create transistor database')
    nch_db = MOSDBDiscrete([nch_config])
    pch_db = MOSDBDiscrete([pch_config])

    design_amp(amp_specs, nch_db, pch_db)


if __name__ == '__main__':
    run_main()
