from compile import *
from utils import cfg_logging_handler

from crnverifier import crn_bisimulation_test
from crnverifier.utils import parse_crn
from dsdobjects import ReactionS
import dsdobjects.objectio as dsdio

import logging
from typing import TypeAlias 
import subprocess
import time
from enum import Enum
import pprint

logger = logging.getLogger(__name__)


class FuelFilterMode(Enum):
    FILTER_FUELS_ONLY = "filter_fuels_only"  # Filter out only species with "fuel" in their name
    FILTER_ALL_SUPPORT = "filter_all_support"  # Filter out all supporting species (keep signal only)
    NO_FILTER = "no_filter"  # Include all species in formal CRN


def run_enumerator(dsd_path: str, vmax=4) -> str:
    out_path = os.path.splitext(dsd_path)[0] + f"_ENUM-CRNcond_vmax{vmax}.pil"

    if os.path.exists(out_path):
        logger.info(f"Enumeration output already exists at {out_path}, skipping enumeration")
        return out_path
    
    # TODO: enumerate.py should take output file as argument
    try:
        result = subprocess.run(
            ["python", "../src/enumerate.py", dsd_path, "-dc", "--max-complex-size", str(vmax)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Enumeration completed successfully for {dsd_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Enumeration failed with return code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("enumerate.py not found or Python not in PATH")
        raise

    return out_path


CRNList: TypeAlias = list[list[str]]


def to_list_format(crn: list[ReactionS]) -> CRNList:
    return [[list(cpx.name for cpx in r.reactants), list(cpx.name for cpx in r.products)]
            for r in crn]


def check_equivalence(fcrn: CRNList, icrn: CRNList, filter_mode: FuelFilterMode,
                      circ: DSDCircuit, condensed: bool, autointerp_intermeds: bool = True) -> bool:
    if filter_mode != FuelFilterMode.NO_FILTER:
        # filter out species from implementation CRN based on mode
        filtered_icrn = []
        exclude_condition = (lambda s: "fuel" in s.lower()) if filter_mode == FuelFilterMode.FILTER_FUELS_ONLY \
                       else (lambda s: s in circ._supporting_species)
        
        for reac, prod in icrn:
            # NOTE: conceptual mis-use of class FormalReaction here for easy filtering
            filtered_icrn.append(FormalReaction(reac, prod).excluding(exclude_condition).list_format())

        icrn = filtered_icrn


    partial_interp = circ.partial_species_interpretation()

    if autointerp_intermeds and filter_mode != FuelFilterMode.FILTER_ALL_SUPPORT:
        # add trivial interpretations of (non-filtered) supporting species
        for s in circ._supporting_species:
            if filter_mode == FuelFilterMode.FILTER_FUELS_ONLY and "fuel" in s.lower():
                continue

            partial_interp[s] = [s]  # trivial interpretation

    logger.debug("initial (partial) interpretation: " + str(partial_interp))

    # logger.info(fcrn); logger.info(icrn) # pprint.format?
    num_formal_species = len(set(s for r in fcrn for s in r[0] + r[1]))

    filter_desc = {
        FuelFilterMode.NO_FILTER: "with all species",
        FuelFilterMode.FILTER_ALL_SUPPORT: "without supporting species",
        FuelFilterMode.FILTER_FUELS_ONLY: "without fuel species only"
    }


    logger.info(f"checking bisimulation equivalence of formal CRN (spc={num_formal_species},reac={len(fcrn)}) "
                f"({filter_desc[filter_mode]}) "
                f"and {len(icrn)} {'condensed' if condensed else 'detailed'} implementation reactions")
    

    t0 = time.time()
    v, full_interp = crn_bisimulation_test(icrn=icrn, fcrn=fcrn, 
                                           interpretation=partial_interp, permissive="default")
    logger.info(f"bisimulation check took {time.time() - t0} sec")
    
    if not v:
        print("NOT equivalent")
    else:
        print("EQUIVALENT")
        logger.info(f"full interpretation:\n{pprint.pformat(full_interp)}")

    return v


def verify_circuit(circ: DSDCircuit, filter_mode: FuelFilterMode, decompose_cycles: bool,
                    dsd_pil_path: str) -> bool:
    logger.info(f"catalytic cycles represented as {'several formals' if decompose_cycles else 'single formal'}")
    include_supporting = (filter_mode != FuelFilterMode.FILTER_ALL_SUPPORT)
    fcrn = [r.list_format() for r in 
            circ.formal_CRN(include_supporting_species=include_supporting, decompose_cycles=decompose_cycles)]

    logger.debug("Formal CRN: " + "; ".join([str(r) for r in fcrn]))

    circ.export_PIL(output_file=dsd_pil_path, name_intermediates=True) # must name intermediates!
    enumCRN_path = run_enumerator(dsd_pil_path)

    icrn_data = dsdio.read_pil(enumCRN_path, is_file=True)
    con_icrn = to_list_format(icrn_data["con_reactions"])
    logger.debug("Condensed impCRN: " + "; ".join([str(r) for r in con_icrn]))

    return check_equivalence(fcrn, con_icrn, filter_mode, circ, condensed=True)
    
    # det_icrn = to_list_format(icrn_data["det_reactions"])
    # check_equivalence(fcrn, det_icrn, filter_mode, circ, condensed=False)


