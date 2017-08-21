# -*- coding: utf-8 -*-


import pprint

from bag.io import read_yaml

from ckt_dsn_ec.mos.core import MOSDBDiscrete
from ckt_dsn_ec.analog.amplifier.opamp_two_stage import OpAmpTwoStage


def run_main():
    w_list = [2]
    nch_conf_list = ['data/mos_char_nch_stack_w2/specs.yaml',
                     # 'data/mos_char_nch_stack/specs.yaml',
                     ]
    pch_conf_list = ['data/mos_char_pch_stack_w2_vbs/specs.yaml',
                     # 'data/mos_char_pch_stack/specs.yaml',
                     ]
    amp_specs_fname = 'dsn_specs/opamp_two_stage.yaml'

    amp_specs = read_yaml(amp_specs_fname)

    print('create transistor database')
    nch_db = MOSDBDiscrete(w_list, nch_conf_list, 1)
    pch_db = MOSDBDiscrete(w_list, pch_conf_list, 1)

    print('create design')
    dsn = OpAmpTwoStage(nch_db, pch_db)
    print('run design')
    dsn.design(**amp_specs)

    print('corners: ', nch_db.env_list)
    pprint.pprint(dsn.get_dsn_info(), width=120)


def run_test(method='linear'):
    w_list = [2]
    nch_conf_list = ['data/mos_char_nch_stack_w2_vbs/specs.yaml',
                     # 'data/mos_char_nch_stack/specs.yaml',
                     ]
    pch_conf_list = ['data/mos_char_pch_stack_w2_vbs/specs.yaml',
                     # 'data/mos_char_pch_stack/specs.yaml',
                     ]

    print('create transistor database')
    nch_db = MOSDBDiscrete(w_list, nch_conf_list, 1, method=method)
    pch_db = MOSDBDiscrete(w_list, pch_conf_list, 1, method=method)

    nch_db.env_list = ['ff_hot']
    pch_db.env_list = ['ff_hot']

    vdd = 0.9
    vin = 0.45
    vtail = 0.6392
    vmid = 0.2614

    pch_db.set_dsn_params(w=2, intent='ulvt', stack=4)
    in_params = pch_db.query(vbs=vdd-vtail, vds=vmid-vtail, vgs=vin-vtail)
    nch_db.set_dsn_params(w=2, intent='svt', stack=2)
    diode_params = nch_db.query(vbs=0, vds=vmid, vgs=vmid)
    nch_db.set_dsn_params(w=2, intent='svt', stack=4)
    ngm_params = nch_db.query(vbs=0, vds=vmid, vgs=vmid)

    gmi = in_params['gm']
    gdsi = in_params['gds']
    gmd = diode_params['gm']
    gdsd = diode_params['gds']
    gmn = ngm_params['gm']
    gdsn = ngm_params['gds']
    print('gmi = %.4g' % gmi)
    print('gdsi = %.4g' % gdsi)
    print('gmd = %.4g' % gmd)
    print('gdsd = %.4g' % gdsd)
    print('gmn = %.4g' % gmn)
    print('gdsn = %.4g' % gdsn)

    print(4 * gmi / (4 * gdsi + 2 * gdsd + 4 * gdsn + 2 * gmd - 4 * gmn))

if __name__ == '__main__':
    # run_main()
    run_test(method='spline')
