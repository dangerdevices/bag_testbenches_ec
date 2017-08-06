# -*- coding: utf-8 -*-


from bag.io import read_yaml

from ckt_dsn_ec.mos.core import MOSCharSS, MOSDB
from ckt_dsn_ec.analog.amplifier.components import LoadDiodePFB, InputGm


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
    nch_config = 'mos_char_specs/mos_char_nch_stack2.yaml'
    pch_config = 'mos_char_specs/mos_char_pch_stack2.yaml'
    load_specs = 'dsn_specs/load_diode_pfb.yaml'
    gm_specs = 'dsn_specs/input_gm.yaml'

    noise_fstart = 20e3
    noise_fstop = noise_fstart + 500
    noise_scale = 1.0
    noise_temp = 310

    load_specs = read_yaml(load_specs)
    gm_specs = read_yaml(gm_specs)

    print('create transistor database')
    nch_sim = MOSCharSS(None, nch_config)
    nch_db = MOSDB(nch_sim, noise_fstart, noise_fstop, noise_scale=noise_scale, noise_temp=noise_temp)
    pch_sim = MOSCharSS(None, pch_config)
    pch_db = MOSDB(pch_sim, noise_fstart, noise_fstop, noise_scale=noise_scale, noise_temp=noise_temp)
    print('create design class')
    load_dsn = LoadDiodePFB(nch_db)
    gm_dsn = InputGm(pch_db)

    print('design load')
    load_dsn.design(**load_specs)
    load_info = load_dsn.get_dsn_info()
    print('load info:')
    print_dsn_info(load_info)

    gm_specs['vd_list'] = load_info['vgs']
    gm_specs['rload_list'] = load_info['ro']
    gm_specs['stack_list'] = [load_info['stack2']]

    print('design gm')
    gm_dsn.design(**gm_specs)
    gm_info = gm_dsn.get_dsn_info()
    print('gm info:')
    print_dsn_info(gm_info)

    print('done')
