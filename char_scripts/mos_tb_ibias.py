# -*- coding: utf-8 -*-

import yaml
import itertools

from bag import float_to_si_string
from bag.core import BagProject
from bag.layout import RoutingGrid, TemplateDB

from abs_templates_ec.mos_char import Transistor


def make_tdb(prj, specs):
    target_lib = specs['impl_lib']

    grid_specs = specs['routing_grid']
    layers = grid_specs['layers']
    spaces = grid_specs['spaces']
    widths = grid_specs['widths']
    bot_dir = grid_specs['bot_dir']

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def generate_lay(prj, specs, sch_params, cell_name):
    temp_db = make_tdb(prj, specs)

    layout_params = specs['layout_params'].copy()
    layout_params['mos_type'] = sch_params['mos_type']
    layout_params['lch'] = sch_params['lch']
    layout_params['w'] = sch_params['w']
    layout_params['threshold'] = sch_params['intent']
    layout_params['stack'] = sch_params['stack']
    layout_params['fg'] = sch_params['nf']
    layout_params['fg_dum'] = sch_params['ndum']

    temp_list = [temp_db.new_template(params=layout_params, temp_cls=Transistor, debug=False), ]
    print('create layout')
    temp_db.batch_layout(prj, temp_list, [cell_name])


def generate_sch(prj, specs, sch_params, dsn_cell_name, tb_cell_name):
    tb_lib = 'bag_ec_testbenches'
    tb_cell = 'mos_tb_ibias'
    dut_lib = 'bag_ec_testbenches'
    dut_cell = 'mos_analogbase'

    impl_lib = specs['impl_lib']

    print('create DUT module')
    dsn = prj.create_design_module(dut_lib, dut_cell)
    print('design DUT')
    dsn.design(**sch_params)
    print('create DUT schematic')
    dsn.implement_design(impl_lib, top_cell_name=dsn_cell_name, erase=True)

    print('create TB module')
    tb_sch = prj.create_design_module(tb_lib, tb_cell)
    print('design TB')
    tb_sch.design(dut_lib=impl_lib, dut_cell=dsn_cell_name)
    print('create TB schematic')
    tb_sch.implement_design(impl_lib, top_cell_name=tb_cell_name)


def characterize(prj, specs):
    sch_params = specs['sch_params'].copy()
    tb_params = specs['tb_params']
    impl_lib = specs['impl_lib']
    view_name = specs['view_name']
    sim_envs = specs['sim_envs']
    rcx_params = specs['rcx_params']

    var_list = []
    swp_val_list = []
    for var_name, val_list in specs['sweep_params'].items():
        var_list.append(var_name)
        swp_val_list.append(val_list)

    sim_info_list = []
    for combo_list in itertools.product(*swp_val_list):
        dsn_name = 'mos_analogbase'
        tb_name = 'mos_analogbase_tb_sp'
        cur_params = dict(zip(var_list, combo_list))
        for name, val in cur_params.items():
            sch_params[name] = val
            if isinstance(val, str):
                suffix = '_%s_%s' % (name, val)
            elif isinstance(val, int):
                suffix = '_%s_%d' % (name, val)
            else:
                suffix = '_%s_%s' % (name, float_to_si_string(val))

            dsn_name += suffix
            tb_name += suffix

        print('design: %s' % dsn_name)
        generate_sch(prj, specs, sch_params, dsn_name, tb_name)
        """
        generate_lay(prj, specs, sch_params, dsn_name)
        print('running lvs')
        lvs_passed, lvs_log = prj.run_lvs(impl_lib, dsn_name)
        if not lvs_passed:
            raise Exception('oops lvs died.  See LVS log file %s' % lvs_log)
        print('lvs passed')

        print('running rcx')
        rcx_passed, rcx_log = prj.run_rcx(impl_lib, dsn_name, rcx_params=rcx_params)
        if not rcx_passed:
            raise Exception('oops rcx died.  See RCX log file %s' % rcx_log)
        print('rcx passed')
        """
        print('create testbench')
        tb = prj.configure_testbench(impl_lib, tb_name)
        for key, val in tb_params.items():
            tb.set_parameter(key, val)
        tb.set_simulation_environments(sim_envs)
        tb.set_simulation_view(impl_lib, dsn_name, view_name)
        tb.update_testbench()
        print('start simulation')
        sim_id = tb.run_simulation(sim_tag=tb_name, block=False)
        sim_info_list.append((tb_name, cur_params, sim_id))

    tb_results = []
    for tb_name, cur_params, sim_id in sim_info_list:
        print('wait for simulation of %s to finish' % tb_name)
        save_dir, retcode = prj.sim.wait(sim_id)
        print('simulation done.')
        if retcode is not None:
            tb_results.append((tb_name, cur_params, save_dir))
        else:
            tb_results.append((tb_name, cur_params, None))

    print('characterization done.')
    return tb_results

if __name__ == '__main__':

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

    characterize(bprj, block_specs)
