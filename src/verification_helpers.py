from compile import *
from utils import cfg_logging_handler

from crnverifier import crn_bisimulation_test, integrated_hybrid_test, compositional_hybrid_test
from crnverifier.utils import parse_crn
from dsdobjects import ReactionS
import dsdobjects.objectio as dsdio

import logging
from typing import TypeAlias 
import subprocess
import time
import pprint

logger = logging.getLogger(__name__)



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

def filter_CRN(crn: CRNList, filter_mode: CRNFilterMode, circ: DSDCircuit) -> CRNList:
    """filter out species from implementation CRN based on mode"""
    if filter_mode == CRNFilterMode.NO_FILTER:
        return crn
    
    filtered_crn = []
    exclude_condition = (lambda s: "fuel" in s.lower()) if filter_mode == CRNFilterMode.REMOVE_FUELS_ONLY \
                    else (lambda s: s in circ._supporting_species)
    
    for reac, prod in crn:
        # NOTE: conceptual mis-use of class FormalReaction here for easy filtering
        filtered_crn.append(FormalReaction(reac, prod).excluding(exclude_condition).list_format())

    return filtered_crn


def check_equivalence(fcrn: CRNList, icrn: CRNList, filter_mode: CRNFilterMode, circ: DSDCircuit) -> bool:
    fcrn = filter_CRN(fcrn, filter_mode, circ)
    icrn = filter_CRN(icrn, filter_mode, circ)
    partial_interp = circ.partial_species_interpretation(filter_mode=filter_mode)
    # assert all and only the formal species are auto-interpreted (does BSE checker assert that?)
    interpreted_values = set(sum(partial_interp.values(), [])) # flatten list of lists
    all_formal_species = set(s for r in fcrn for s in r[0] + r[1])
    assert interpreted_values == all_formal_species, f"Partial interpretation should cover exactly the formal species. Got {interpreted_values}, expected {all_formal_species}"


    logger.debug("initial (partial) interpretation: " + str(partial_interp))
    # logger.info(fcrn); logger.info(icrn) # pprint.format?
    num_formal_species = len(set(s for r in fcrn for s in r[0] + r[1]))
    logger.info(f"checking bisimulation equivalence of formal CRN (spc={num_formal_species},reac={len(fcrn)}) "
                f"and {len(icrn)} implementation reactions")


    t0 = time.time()
    v, full_interp = crn_bisimulation_test(icrn=icrn, fcrn=fcrn, interpretation=partial_interp, permissive="default")
    # v, full_interp = compositional_hybrid_test(fcrn, icrn, all_formal_species, partial_interp)
    # v, full_interp = integrated_hybrid_test(fcrn, icrn, all_formal_species, partial_interp)
    logger.info(f"bisimulation check took {time.time() - t0} sec")
    
    if not v:
        print("NOT equivalent")
    else:
        print("EQUIVALENT")
        logger.info(f"full interpretation:\n{pprint.pformat(full_interp)}")

    return v


def verify_circuit(circ: DSDCircuit, dsd_pil_path: str, filter_mode: CRNFilterMode, decompose_cycles: bool=False) -> bool:
    """
    Verify that the given DSD circuit's formal CRN is bisimulation equivalent to the implementation CRN
    obtained from enumeration. 
    Arguments:
    - circ: DSDCircuit object representing the circuit to verify    
    - dsd_pil_path: file path to save the exported circuit for enumeration
    - filter_mode: FuelFilterMode enum value specifying how to filter species from the CRNs before equivalence checking
    - decompose_cycles: whether to represent catalytic cycles (eg. signal restoration) 
        as multiple reactions (True) or single reactions (False)
    """
    logger.info(f"catalytic cycles represented as {'several formals' if decompose_cycles else 'single formal'}")
    fcrn = [r.list_format() for r in circ.formal_CRN(decompose_cycles=decompose_cycles)]

    logger.debug("unfiltered Formal CRN: " + "; ".join([str(r) for r in fcrn]))

    circ.export_PIL(output_file=dsd_pil_path, name_intermediates=True) # must name intermediate signals!
    enumCRN_path = run_enumerator(dsd_pil_path)

    icrn_data = dsdio.read_pil(enumCRN_path, is_file=True)
    con_icrn = to_list_format(icrn_data["con_reactions"])
    logger.debug("unfiltered Condensed impCRN: " + "; ".join([str(r) for r in con_icrn]))

    return check_equivalence(fcrn, con_icrn, filter_mode, circ)
    
    # det_icrn = to_list_format(icrn_data["det_reactions"])
    # check_equivalence(fcrn, det_icrn, filter_mode, circ, condensed=False)


