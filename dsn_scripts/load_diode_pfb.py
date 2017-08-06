# -*- coding: utf-8 -*-


from bag.io import read_yaml

from ckt_dsn_ec.mos.core import MOSCharSS, MOSDB
from ckt_dsn_ec.analog.amplifier.components import LoadDiodePFB


def print_dsn_info(info):
    if info is None:
        print('No solution found')
    else:
        for key, val in info.items():
            if isinstance(val, list):
                print('%s = [%s]' % (key, ', '.join(('%.3g' % v for v in val))))
            elif isinstance(val, str):
                print('%s = %s' % (key, val))
            else:
                print('%s = %.3g' % (key, val))


if __name__ == '__main__':
    config_file = 'mos_char_specs/mos_char_nch_stack_w2.yaml'
    dsn_file = 'dsn_specs/load_diode_pfb.yaml'

    noise_fstart = 20e3
    noise_fstop = noise_fstart + 500
    noise_scale = 1.0
    noise_temp = 310

    dsn_specs = read_yaml(dsn_file)

    print('create transistor database')
    mos_sim = MOSCharSS(None, config_file)
    mos_db = MOSDB(mos_sim, noise_fstart, noise_fstop, noise_scale=noise_scale, noise_temp=noise_temp)
    print('create design class')
    load_dsn = LoadDiodePFB(mos_db)

    print('run design')
    load_dsn.design(**dsn_specs)
    dsn_info = load_dsn.get_dsn_info()
    print_dsn_info(dsn_info)
    print('done')
