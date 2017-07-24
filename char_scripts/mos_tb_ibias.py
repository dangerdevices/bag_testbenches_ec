# -*- coding: utf-8 -*-

import os
import yaml
import itertools

from bag import float_to_si_string
from bag.core import BagProject
from bag.data import load_sim_results, save_sim_results
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


def generate_lay(prj, specs, sch_params, cell_name, temp_db):
    layout_params = specs['layout_params'].copy()
    layout_params['mos_type'] = sch_params['mos_type']
    layout_params['lch'] = sch_params['l']
    layout_params['w'] = sch_params['w']
    layout_params['threshold'] = sch_params['intent']
    layout_params['stack'] = sch_params['stack']
    layout_params['fg'] = sch_params['nf']
    layout_params['fg_dum'] = sch_params['ndum']

    temp_list = [temp_db.new_template(params=layout_params, temp_cls=Transistor, debug=False), ]
    temp_db.batch_layout(prj, temp_list, [cell_name])
    return layout_params


def generate_sch(prj, specs, sch_params, dsn_cell_name, tb_cell_name):
    tb_lib = 'bag_ec_testbenches'
    tb_cell = 'mos_tb_ibias'
    dut_lib = 'bag_ec_testbenches'
    dut_cell = 'mos_analogbase'

    impl_lib = specs['impl_lib']

    dsn = prj.create_design_module(dut_lib, dut_cell)
    dsn.design(**sch_params)
    dsn.implement_design(impl_lib, top_cell_name=dsn_cell_name, erase=True)

    tb_sch = prj.create_design_module(tb_lib, tb_cell)
    tb_sch.design(dut_lib=impl_lib, dut_cell=dsn_cell_name)
    tb_sch.implement_design(impl_lib, top_cell_name=tb_cell_name, erase=True)


def get_tb_dsn_name(dsn_name_base, tb_name_base, var_list, combo_list):
    suffix = ''
    for var, val in zip(var_list, combo_list):
        if isinstance(val, str):
            suffix += '_%s_%s' % (var, val)
        elif isinstance(val, int):
            suffix += '_%s_%d' % (var, val)
        else:
            suffix += '_%s_%s' % (var, float_to_si_string(val))

    return dsn_name_base + suffix, tb_name_base + suffix


def characterize(prj, specs):
    sch_params = specs['sch_params'].copy()
    tb_params = specs['tb_params']
    impl_lib = specs['impl_lib']
    view_name = specs['view_name']
    sim_envs = specs['sim_envs']
    rcx_params = specs['rcx_params']
    results_dir = specs['results_dir']
    dsn_name_base = specs['dsn_name_base']
    tb_name_base = specs['tb_name_base']

    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'specs.yaml'), 'w') as specs_file:
        yaml.dump(specs, specs_file)

    temp_db = make_tdb(prj, specs)

    # get sweep parameters
    swp_par_dict = specs['sweep_params']
    var_list = sorted(swp_par_dict.keys())
    swp_val_list = [swp_par_dict[var] for var in var_list]

    # make schematic, layout, and start LVS jobs
    job_info_list = []
    for combo_list in itertools.product(*swp_val_list):
        dsn_name, tb_name = get_tb_dsn_name(dsn_name_base, tb_name_base, var_list, combo_list)
        for name, val in zip(var_list, combo_list):
            sch_params[name] = val

        print('create schematic/testbench for %s' % dsn_name)
        generate_sch(prj, specs, sch_params, dsn_name, tb_name)
        print('create layout for %s' % dsn_name)
        lay_params = generate_lay(prj, specs, sch_params, dsn_name, temp_db)
        print('start lvs job')
        lvs_id, lvs_log = prj.run_lvs(impl_lib, dsn_name, block=False)
        job_info_list.append([tb_name, dsn_name, lay_params, lvs_id, lvs_log])

    # start RCX jobs
    for idx in range(len(job_info_list)):
        tb_name, dsn_name, lay_params, lvs_id, lvs_log = job_info_list[idx]
        print('wait for %s LVS to finish' % dsn_name)
        lvs_passed = prj.wait_lvs_rcx(lvs_id)
        if not lvs_passed:
            raise Exception('oops lvs died for %s.  See LVS log file %s' % (dsn_name, lvs_log))
        print('lvs passed.  start rcx for %s' % dsn_name)
        rcx_id, rcx_log = prj.run_rcx(impl_lib, dsn_name, block=False, rcx_params=rcx_params)
        job_info_list[idx][3] = rcx_id
        job_info_list[idx][4] = rcx_log

    # configure testbench and start simulations
    for idx in range(len(job_info_list)):
        tb_name, dsn_name, lay_params, rcx_id, rcx_log = job_info_list[idx]
        print('wait for %s RCX to finish' % dsn_name)
        rcx_passed = prj.wait_lvs_rcx(rcx_id)
        if not rcx_passed:
            raise Exception('oops rcx died for %s.  See RCX log file %s' % (dsn_name, rcx_log))
        print('rcx passed.  setup testbench %s' % tb_name)
        tb = prj.configure_testbench(impl_lib, tb_name)
        for key, val in tb_params.items():
            tb.set_parameter(key, val)
        tb.set_simulation_environments(sim_envs)
        tb.set_simulation_view(impl_lib, dsn_name, view_name)
        tb.update_testbench()
        print('start simulation for %s' % tb_name)
        sim_id = tb.run_simulation(sim_tag=tb_name, block=False)
        job_info_list[idx][3] = sim_id
        job_info_list[idx][4] = tb

    for tb_name, dsn_name, lay_params, sim_id, tb in job_info_list:
        print('wait for %s to finish' % tb_name)
        save_dir = tb.wait()
        print('simulation done.')
        if save_dir is not None:
            try:
                cur_results = load_sim_results(save_dir)
            except:
                print('Error when loading results for %s' % tb_name)
                cur_results = None
        else:
            cur_results = None

        cur_result_dir = os.path.join(results_dir, tb_name)
        os.makedirs(cur_result_dir, exist_ok=True)

        info = dict(
            tb_name=tb_name,
            dsn_name=dsn_name,
            save_dir=save_dir,
            lay_params=lay_params,
        )
        with open(os.path.join(cur_result_dir, 'info.yaml'), 'w') as info_file:
            yaml.dump(info, info_file)
        if cur_results is not None:
            save_sim_results(cur_results, os.path.join(cur_result_dir, 'data.hdf5'))

    print('characterization done.')

if __name__ == '__main__':

    config_file = 'mos_char_specs/mos_tb_ibias.yaml'

    with open(config_file, 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    characterize(bprj, block_specs)
