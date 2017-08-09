# -*- coding: utf-8 -*-


from bag.io import read_yaml

from ckt_dsn_ec.mos.core import MOSDBDiscrete
from ckt_dsn_ec.analog.amplifier.opamp_two_stage import OpAmpTwoStage


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
    w_list = [2]
    nch_conf_list = ['data/mos_char_nch_stack_w2/specs.yaml',
                     # 'data/mos_char_nch_stack/specs.yaml',
                     ]
    pch_conf_list = ['data/mos_char_pch_stack_w2_vbs/specs.yaml',
                     # 'data/mos_char_pch_stack/specs.yaml',
                     ]
    amp_specs_fname = 'dsn_specs/opamp_two_stage.yaml'

    noise_fstart = 20e3
    noise_fstop = noise_fstart + 500
    noise_scale = 1.0
    noise_temp = 310

    amp_specs = read_yaml(amp_specs_fname)

    print('create transistor database')
    nch_db = MOSDBDiscrete(w_list, nch_conf_list, 1, noise_fstart, noise_fstop,
                           noise_scale=noise_scale, noise_temp=noise_temp)
    pch_db = MOSDBDiscrete(w_list, pch_conf_list, 1, noise_fstart, noise_fstop,
                           noise_scale=noise_scale, noise_temp=noise_temp)

    print('create design')
    dsn = OpAmpTwoStage(nch_db, pch_db)
    print('run design')
    dsn.design(**amp_specs)

    print_dsn_info(dsn.get_dsn_info())
