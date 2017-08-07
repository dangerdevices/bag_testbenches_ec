# -*- coding: utf-8 -*-

"""This module contains design algorithm for a traditional two stage operational amplifier."""

from typing import List, Optional, Dict, Any

from ckt_dsn_ec.mos.core import MOSDBDiscrete

from .components import LoadDiodePFB, InputGm


class OpAmpTwoStage(object):
    """A two stage fully differential operational amplifier.

    The first stage is a differential amplifier with diode + positive feedback load, the
    second stage is a psuedo-differential common source amplifier.

    This topology has the following advantages:
    1. large output swing.
    2. Common mode feedback is only required for the second stage.
    """

    def __init__(self, nch_db, pch_db):
        # type: (MOSDBDiscrete, MOSDBDiscrete) -> None
        self._nch_db = nch_db
        self._pch_db = pch_db
        self._amp_info = None

    def design(self,
               itarg_list,  # type: List[float]
               vg_list,  # type: List[float]
               l,  # type: float
               vstar_gm_min,  # type: float
               vstar_load_min,  # type: float
               vds_tail_min,  # type: float
               seg_gm_min,  # type: int
               vdd,  # type: float
               pmos_input=True,  # type: bool
               ):
        if pmos_input:
            load = LoadDiodePFB(self._nch_db)
            gm = InputGm(self._pch_db)
        else:
            load = LoadDiodePFB(self._pch_db)
            gm = InputGm(self._nch_db)

        # design load
        load.design(itarg_list, vstar_load_min, l)
        load_info = load.get_dsn_info()
        rload_list = load_info['ro']
        cload_list = load_info['co']
        stack_ngm = load_info['stack_ngm']
        if pmos_input:
            vd_list = load_info['vgs']
            vb = vdd
        else:
            vd_list = [vdd - vgs for vgs in load_info['vgs']]
            vb = 0

        # design input gm
        gm.design(itarg_list, vg_list, vd_list, rload_list, vb, vstar_gm_min, vds_tail_min, l,
                  seg_min=seg_gm_min, stack_list=[stack_ngm])
        gm_info = gm.get_dsn_info()
        gm1_list = gm_info['gm']
        cdd_gm = gm_info['cdd']
        ro_gm = gm_info['ro']

        ro1_list = [1 / (1/rogm + 1/rol) for rogm, rol in zip(ro_gm, rload_list)]
        gain1_list = [g * r for g, r in zip(gm1_list, ro1_list)]
        c1_list = [cl + cg for cl, cg in zip(cload_list, cdd_gm)]

        self._amp_info = dict(
            vtail=gm_info['vs'],
            vmid=vd_list,

            vstar=gm_info['vstar'],
            cin=gm_info['cgg'],
            gm1=gm1_list,
            ro1=ro1_list,
            gain1=gain1_list,
            c1=c1_list,

            w_gm=gm_info['w'],
            intent_gm=gm_info['intent'],
            seg_gm=gm_info['seg'],
            stack_gm=gm_info['stack'],

            w_load=load_info['w'],
            intent_load=load_info['intent'],
            seg_diode=load_info['seg_diode'],
            seg_ngm=load_info['seg_ngm'],
            stack_diode=load_info['stack_diode'],
            stack_ngm=load_info['stack_ngm'],
        )

    def get_dsn_info(self):
        # type: () -> Optional[Dict[str, Any]]
        return self._amp_info
