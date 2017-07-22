# -*- coding: utf-8 -*-

import yaml

from bag.core import BagProject


def generate_sch(prj, specs):
    tb_lib = 'bag_ec_testbenches'
    tb_cell = 'mos_tb_ibias'
    dut_lib = 'bag_ec_testbenches'
    dut_cell = 'mos_analogbase'

    dut_params = specs['dut_params']

    print('create DUT module')
    dsn = prj.create_design_module(dut_lib, dut_cell)
    print('design DUT')
    dsn.design(**dut_params)
    print('create DUT schematic')
    dsn.implement_design(impl_lib, top_cell_name=dut_cell, erase=True)

    print('create TB module')
    tb_sch = prj.create_design_module(tb_lib, tb_cell)
    print('design TB')
    tb_sch.design(dut_lib=impl_lib, dut_cell=dut_cell)
    print('create TB schematic')
    tb_sch.implement_design(impl_lib, top_cell_name=tb_cell)


if __name__ == '__main__':

    impl_lib = 'MOS_CHAR'
    spec_file = 'mos_char_specs/mos_tb_ibias.yaml'

    with open(spec_file, 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    generate_sch(bprj, block_specs)
