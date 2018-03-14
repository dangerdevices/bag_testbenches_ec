# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sciopt
import scipy.signal as scisig

import matplotlib.pyplot as plt

from bag.data.lti import LTICircuit

from verification_ec.mos.query import MOSDBDiscrete


def get_db(nch_dir, pch_dir, intent='standard', interp_method='spline', sim_env='tt'):
    env_list = [sim_env]

    nch_db = MOSDBDiscrete([nch_dir], interp_method=interp_method)
    pch_db = MOSDBDiscrete([pch_dir], interp_method=interp_method)

    nch_db.env_list = pch_db.env_list = env_list
    nch_db.set_dsn_params(intent=intent)
    pch_db.set_dsn_params(intent=intent)

    return nch_db, pch_db


def tf_vs_cfb(op_in, op_load, op_tail, cload, fg=2):
    cmin = 5e-16
    cmax = 5e-14
    fmin = 6
    fmax = 11
    num_c = 5
    num_f = 1000

    scale_load = op_in['ibias'] / op_load['ibias'] * fg

    cir = LTICircuit()
    cir.add_transistor(op_in, 'mid', 'in', 'gnd', 'gnd', fg=fg)
    cir.add_transistor(op_load, 'mid', 'gnd', 'gnd', 'gnd', fg=scale_load)
    cir.add_transistor(op_load, 'out', 'mid', 'gnd', 'gnd', fg=scale_load)
    cir.add_transistor(op_tail, 'out', 'gnd', 'gnd', 'gnd', fg=fg)
    cir.add_cap(cload, 'out', 'gnd')

    cfb = np.logspace(np.log10(cmin), np.log10(cmax), num_c).tolist()
    gfb = op_load['gm'] * scale_load
    cir.add_conductance(gfb, 'mid', 'x')

    print('fg_in = %d, fg_load=%.3g, rfb = %.4g' % (fg, scale_load, 1/gfb))

    fvec = np.logspace(fmin, fmax, num_f)
    wvec = 2 * np.pi * fvec
    plt.figure(1)
    plt.plot(fvec, [0] * len(fvec), '--k')
    for cval in cfb:
        cir.add_cap(cval, 'x', 'out')
        num, den = cir.get_num_den('in', 'out')
        cir.add_cap(-cval, 'x', 'out')

        _, mag, phase = scisig.bode((num, den), w=wvec)
        poles = np.sort(np.abs(np.poly1d(den).roots) / (2 * np.pi))
        print(poles)
        poles = poles[:2]
        mag_poles = np.interp(poles, fvec, mag)

        p = plt.semilogx(fvec, mag, label='$C_{f} = %.3g$f' % (cval * 1e15))
        color = p[0].get_color()
        plt.plot(poles, mag_poles, linestyle='', color=color, marker='o')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.show()


def run_main():
    nch_dir = 'data/nch_w4'
    pch_dir = 'data/pch_w4'
    intent = 'ulvt'

    vtail = 0.15
    vdd = 0.9
    vmid = vdd / 2

    cload = 10e-15

    nch_db, pch_db = get_db(nch_dir, pch_dir, intent=intent)

    op_in = nch_db.query(vbs=-vtail, vds=vmid-vtail, vgs=vmid-vtail)
    op_load = pch_db.query(vbs=0, vds=vmid-vdd, vgs=vmid-vdd)
    in_ibias = op_in['ibias']

    ibias_fun = nch_db.get_function('ibias')

    def fun_zero(vg):
        arg = nch_db.get_fun_arg(vgs=vg, vds=vtail, vbs=0)
        return (ibias_fun(arg) - in_ibias) * 1e6

    vbias = sciopt.brentq(fun_zero, 0, vdd)
    # noinspection PyTypeChecker
    op_tail = nch_db.query(vbs=0, vds=vtail, vgs=vbias)

    tf_vs_cfb(op_in, op_load, op_tail, cload)


if __name__ == '__main__':
    run_main()
