# -*- coding: utf-8 -*-

"""This script designs a simple diff amp with gain/bandwidth spec for BAG CICC paper."""


from bag.io import read_yaml

from ckt_dsn_ec.mos.core import MOSDBDiscrete
from ckt_dsn_ec.analog.amplifier.components import InputGm


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
    nch_config = 'mos_char_specs/nch_w4_amp.yaml'
    pch_config = 'mos_char_specs/pch_w4_amp.yaml'
    amp_specs = 'specs_dsn/diffamp_paper.yaml'

    amp_specs = read_yaml(amp_specs)

    print('create transistor database')
    nch_db = MOSDBDiscrete([pch_config])
    print('create design class')
    gm_dsn = InputGm(pch_db)

    print('design gm')
    gm_dsn.design(**gm_specs)
    gm_info = gm_dsn.get_dsn_info()
    print('gm info:')
    print_dsn_info(gm_info)

    print('done')
