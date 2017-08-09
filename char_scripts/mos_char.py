# -*- coding: utf-8 -*-

from bag.core import BagProject

from ckt_dsn_ec.mos.core import MOSCharSS

if __name__ == '__main__':

    config_file = 'mos_char_specs/mos_char_pch_stack_w2_vbs.yaml'

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    sim = MOSCharSS(bprj, config_file)

    # sim.run_lvs_rcx(tb_type='tb_ibias')
    # sim.run_simulations('tb_ibias')
    # sim.process_ibias_data()

    # sim.run_simulations('tb_sp')
    sim.run_simulations('tb_noise', overwrite=False)

    """
    fc = 100e3
    fbw = 500
    vgd_opt = 0.0
    temperature = 310
    inorm = 1e-6
    dname = 'MOS_PCH_STACK_intent_svt_l_90n'
    sim.plot_dsn_info(dname, fc - fbw / 2, fc + fbw / 2, vgd_opt, itarg=inorm, temp=temperature)
    """
