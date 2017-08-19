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


if __name__ == '__main__':
    run_main()
