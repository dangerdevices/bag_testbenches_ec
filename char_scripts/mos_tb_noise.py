# -*- coding: utf-8 -*-

import os
import itertools

import yaml
import numpy as np

from bag import float_to_si_string
from bag.core import BagProject
from bag.data import load_sim_results, save_sim_results


def read_yaml(fname):
    with open(fname, 'r') as f:
        return yaml.load(f)


def generate_sch(prj, specs, dsn_cell_name, tb_cell_name):
    tb_lib = 'bag_ec_testbenches'
    tb_cell = 'mos_tb_noise'

    impl_lib = specs['impl_lib']

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
    results_dir = specs['results_dir']
    vgs_file = specs['vgs_file']
    impl_lib = specs['impl_lib']
    view_name = specs['view_name']
    dsn_name_base = specs['dsn_name_base']
    tb_name_base = specs['tb_name_base']

    swp_par_dict = specs['sweep_params']
    tb_params = specs['tb_params']
    tb_sweep = specs['tb_sweep']
    sim_envs = specs['sim_envs']

    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'specs.yaml'), 'w') as specs_file:
        yaml.dump(specs, specs_file)

    # get sweep parameters
    var_list = sorted(swp_par_dict.keys())
    swp_val_list = [swp_par_dict[var] for var in var_list]

    vgs_info = read_yaml(vgs_file)

    # make schematic and start simulation jobs
    job_info_list = []
    vds_start = tb_sweep['vds_start']
    vds_num = tb_sweep['vds_num']
    for combo_list in itertools.product(*swp_val_list):
        cur_tb_params = tb_params.copy()
        dsn_name, tb_name = get_tb_dsn_name(dsn_name_base, tb_name_base, var_list, combo_list)
        vgs_min, vgs_max = vgs_info[dsn_name]
        vgs_num = tb_sweep['vgs_num']

        print('create testbench for %s' % dsn_name)
        generate_sch(prj, specs, dsn_name, tb_name)

        tb = prj.configure_testbench(impl_lib, tb_name)

        for key, val in cur_tb_params.items():
            tb.set_parameter(key, val)

        vds_end = vgs_min if abs(vgs_min) > abs(vgs_max) else vgs_max

        if vds_start < vds_end:
            vds_vals = np.linspace(vds_start, vds_end, vds_num)
        else:
            vds_vals = np.linspace(vds_end, vds_start, vds_num)
        # S parameter sweep adds 1 to vgs_num
        vgs_vals = np.linspace(vgs_min, vgs_max, vgs_num + 1)
        tb.set_sweep_parameter('vds', values=vds_vals)
        tb.set_sweep_parameter('vbs', values=tb_sweep['vbs'])
        tb.set_sweep_parameter('vgs', values=vgs_vals)
        tb.set_simulation_environments(sim_envs)
        tb.set_simulation_view(impl_lib, dsn_name, view_name)
        tb.update_testbench()
        print('start simulation for %s' % tb_name)
        sim_id = tb.run_simulation(sim_tag=tb_name, block=False)
        job_info_list.append((dsn_name, tb_name, tb, sim_id))

    for dsn_name, tb_name, tb, sim_id in job_info_list:
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
        )
        with open(os.path.join(cur_result_dir, 'info.yaml'), 'w') as info_file:
            yaml.dump(info, info_file)
        if cur_results is not None:
            save_sim_results(cur_results, os.path.join(cur_result_dir, 'data.hdf5'))

    print('characterization done.')

if __name__ == '__main__':

    config_file = 'mos_char_specs/mos_tb_noise_pch_stack.yaml'
    block_specs = read_yaml(config_file)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    characterize(bprj, block_specs)
    # process_data(block_specs['results_dir'], vgs_file)
