"""Enumerate reactions Peppercorn, set timescale separation for cooperative hybridization"""
import os
import time
import sys
import argparse
from typing import Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

from utils import read_metadata, print_CRN_stats, cfg_logging_handler

#from peppercornenumerator.peppercorn import add_peppercorn_args, main as peppercorn_main
from peppercornenumerator.peppercorn import *# ColorFormatter, add_peppercorn_args, set_handle_verbosity
from peppercornenumerator import __version__ as ppcorn_version
from peppercornenumerator.enumerator import PolymerizationError
from peppercornenumerator.output import write_vdsd, write_sbml

import dsdobjects.objectio as dsdio # WARNING: don't remove package alias (name clashes with peppercorn catch-all import)

# Configure logging for this module
logger = logging.getLogger(__name__)


class ReactionType(Enum):
    OPENING = auto()
    MIGRATION_3WAY = auto()
    
def canonical_direct_3way_migration_rate(domain_len: int) -> float:
    """Calculate the canonical 3-way migration rate for a given domain length"""
    return 1/3 * 10**3 / domain_len


class TimescaleSeparation(ABC):
    """Base class for timescale separation parameters"""
    
    @abstractmethod
    def enables_reaction(self, reaction_type: ReactionType, domain_len: int, **kwargs) -> bool:
        """check if the reaction is enabled by the timescale separation"""
        pass

    def disables_reaction(self, reaction_type: ReactionType, domain_len: int, **kwargs) -> bool:
        """check if the reaction is disabled by the timescale separation"""
        return not self.enables_reaction(reaction_type, domain_len, **kwargs)


@dataclass
class RateIndependentTimescaleSep(TimescaleSeparation):
    """timescale separation parameters for enumerating unimolecular reactions"""
    # NOTE: no support for 1-1, 1-2 specific cutoffs

    release_cutoff: int
    """release cutoff for all opening reactions"""

    def enables_reaction(self, reaction_type: ReactionType, domain_len: int, **kwargs) -> bool:
        match reaction_type:
            case ReactionType.OPENING:
                return domain_len <= self.release_cutoff
            case ReactionType.MIGRATION_3WAY:
                return True # always enabled


class ReactionSpeed(Enum):
    """Reaction types for unimolecular reactions"""
    NEGLIGIBLE = auto()
    SLOW = auto()
    FAST = auto()

@dataclass
class RateDependentTimescaleSep(TimescaleSeparation):
    """timescale separation parameters for enumerating unimolecular reactions"""
    k_slow: float
    """minimum rate constant for slow reactions"""
    k_fast: float
    """maximum rate constant for fast reactions"""

    def __post_init__(self):
        if not (self.k_slow <= self.k_fast):
            raise ValueError("k_slow <= k_fast required")
        if self.k_slow < 0:
            raise ValueError("k_slow must be non-negative")

    def reaction_speed_label(self, reaction_type: ReactionType, domain_len: int, **kwargs) -> ReactionSpeed:
        """determine type of a unimolecular reaction based on rate constant"""

        match reaction_type:
            case ReactionType.OPENING:
                uni_rate_const = opening_rate(domain_len, dG_bp=kwargs["dG_bp"])
            case ReactionType.MIGRATION_3WAY:
                uni_rate_const = canonical_direct_3way_migration_rate(domain_len)

        if uni_rate_const < self.k_slow:
            return ReactionSpeed.NEGLIGIBLE
        elif uni_rate_const < self.k_fast:
            return ReactionSpeed.SLOW
        else:
            return ReactionSpeed.FAST

    def enables_reaction(self, reaction_type: ReactionType, domain_len: int, **kwargs) -> bool:
        """check if the reaction is enabled by the timescale separation"""
        return self.reaction_speed_label(reaction_type, domain_len, **kwargs) != ReactionSpeed.NEGLIGIBLE

    def disables_reaction(self, reaction_type: ReactionType, domain_len: int, **kwargs) -> bool:
        """check if the reaction is disabled by the timescale separation"""
        return self.reaction_speed_label(reaction_type, domain_len, **kwargs) == ReactionSpeed.NEGLIGIBLE



def cooperative_hybridization_limits(toehold_len: int, args: argparse.Namespace,
                                     tight_lower_bound=True, epsilon=1e-10) -> RateDependentTimescaleSep:
    """return timescale separation limits for cooperative hybridization
    (requirement for cooperative hybridization: k_slow <= TH opening rate < k_fast)
    
    Args:
        toehold_len (int): length of the toehold that initiates cooperative hybridization
        args (argparse.Namespace): command line arguments
        tight_lower_bound (bool): if True, use a tight lower bound for k_slow (discard as many reactions as possible)
        epsilon (float): tolerance for floating point comparison in classifying reactions
    """
    if args.no_max_helix:
        raise NotImplementedError("timescale separation fix implemented only for max-helix semantics")

    critical_rate = opening_rate(toehold_len, dG_bp=args.dG_bp) # opening rate of the toehold that initiates cooperative hybridization

    # treat toehold opening as slow reaction (so cooperative hybridization can complete) 
    k_fast = critical_rate + epsilon 
    # NOTE: this means opening of any LONGER toehold is also considered slow
    if args.verbose:
        logger.debug(f"toeholds with length >= {toehold_len} can initiate cooperative hybridization")

    # treat opening of longer domains as negligible (avoid funny business with reporters etc.), assuming they are all recognition domains (not toeholds)
    if tight_lower_bound:
        k_slow = critical_rate - epsilon 
    else:
        shortest_recognition_domain = min(...)
        # NOTE: need to also enable all migrations using ["migration_sequence_lengths"] ???
        k_slow = opening_rate(shortest_recognition_domain, dG_bp=args.dG_bp) + epsilon # eps to ensure threshold is above opening rate of recognition domains

    return RateDependentTimescaleSep(k_slow, k_fast)


# below function: slightly modified peppercornenumerator.peppercorn.main()
# original code: https://github.com/DNA-and-Natural-Algorithms-Group/peppercornenumerator/blob/master/peppercornenumerator/peppercorn.py
# original licensed under: The MIT License (MIT); see below
# ---
# Copyright (c) 2014-2020 Stefan Badelt, Casey Grun, Karthik Sarma, Brian Wolfe, Seung Woo Shin and Erik Winfree <winfree@caltech.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ---
def run_peppercorn(args, output_filename: str) -> Tuple[Enumerator, float, float]:
    """Run the Peppercorn reaction enumerator with the given arguments.
    Args:
        args (argparse.Namespace): Command line arguments.
        output_filename (str): Output filename for the enumerated reactions.

    Returns:
        Enumerator: The enumerator object containing the enumerated reactions.
        float: Time taken for the enumeration process in seconds (excluding setup and condensation).
        float: Time taken for the condensation process in seconds.
    """

    # ~~~~~~~~~~~~~
    # Logging Setup 
    # ~~~~~~~~~~~~~
    title = "Peppercorn Domain-level Reaction Enumerator"
    if args.logfile:
        banner = "{} {}".format(title, ppcorn_version)
        # fh = logging.FileHandler(args.logfile)
        # formatter = logging.Formatter('%(levelname)s - %(message)s')
        # set_handle_verbosity(fh, args.verbose)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)
    else:
        banner = "{} {}".format(colors.BOLD + title + colors.ENDC, 
                                  colors.GREEN + ppcorn_version + colors.ENDC)
        # ch = logging.StreamHandler()
        # formatter = ColorFormatter('%(levelname)s %(message)s', use_color = True)
        # set_handle_verbosity(ch, args.verbose)
        # ch.setFormatter(formatter)
        # logger.addHandler(ch)

    logger.info(banner)

    assertions = False
    try:
        assert False
    except AssertionError:
        assertions = True
    logger.debug(f'Using assert statements: {assertions}.')

    systeminput = args.input_filename
    if not systeminput :
        if sys.stdout.isatty():
            logger.info("Reading file from STDIN, Ctrl-D to stop")
        systeminput = ''
        for l in sys.stdin:
            systeminput += l
        if args.interactive:
            logger.error("Interactive mode needs to read input from file, not STDIN.")
            raise SystemExit

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Input parsing to set initial complexes for enumeration #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    try :
        complexes, reactions = read_pil(systeminput, args.input_filename is not None)
    except ParseException as ex_pil:
        try :
            complexes, reactions = read_seesaw(systeminput, 
                    args.input_filename is not None, 
                    conc = args.seesaw_conc, 
                    explicit = args.seesaw_explicit, 
                    reactions = args.seesaw_reactions)
        except ParseException as ex_ssw:
            logger.error('Pil-format parsing error:')
            logger.error('Cannot parse line {:5d}: "{}"'.format(ex_pil.lineno, ex_pil.line))
            logger.error('                          {} '.format(' ' * (ex_pil.col-1) + '^'))
            logger.error('SeeSaw-format parsing error:')
            logger.error('Cannot parse line {:5d}: "{}"'.format(ex_ssw.lineno, ex_ssw.line))
            logger.error('                          {} '.format(' ' * (ex_ssw.col-1) + '^'))
            raise SystemExit


    init_cplxs = [x for x in complexes.values() if x.concentration is None or x.concentration[1] != 0]
    name_cplxs = list(complexes.values())
    enum = Enumerator(init_cplxs, reactions, named_complexes = name_cplxs)

    # Log initial complexes
    logger.info("")
    logger.info("Initial complexes:")
    for c in enum.initial_complexes:
        logger.info("{}: {}".format(c, c.kernel_string))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Transfer options to enumerator object #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    logger.info("")
    logger.info("Enumeration settings:")
    enum.max_complex_size = args.max_complex_size
    logger.info("Max complex size = {}".format(enum.max_complex_size))
    enum.max_complex_count = max(args.max_complex_count, len(complexes))
    logger.info("Max complex count = {}".format(enum.max_complex_count))
    enum.max_reaction_count = max(args.max_reaction_count, len(reactions))
    logger.info("Max reaction count = {}".format(enum.max_reaction_count))
    enum.max_helix = not args.no_max_helix
    logger.info('Max-helix semantics = {}'.format(enum.max_helix))
    enum.reject_remote = args.reject_remote 
    logger.info('Reject-remote semantics = {}'.format(enum.reject_remote))
    enum.dG_bp = args.dG_bp
    logger.info('Average strength of a toehold base-pair dG_bp = {}'.format(enum.dG_bp))
    if args.ignore_branch_3way:
        if branch_3way in UNI_REACTIONS:
            UNI_REACTIONS.remove(branch_3way)
        logger.info('No 3-way branch migration.')
    if args.ignore_branch_4way:
        if branch_4way in UNI_REACTIONS:
            UNI_REACTIONS.remove(branch_4way)
        logger.info('No 4-way branch migration.')
    PepperComplex.PREFIX = args.complex_prefix
    logger.info("Prefix for new complexes = {PepperComplex.PREFIX}")

    # Set either k-slow or release cutoff.
    if args.k_slow:
        if args.release_cutoff is not None:
            args.release_cutoff = None
            logger.warning('Release-cutoff overwritten by k-slow!')
        if args.release_cutoff_1_1 != args.release_cutoff_1_2:
            logger.warning('Release-cutoff (1,1) overwritten by k-slow!')
            logger.warning('Release-cutoff (1,2) overwritten by k-slow!')
        rc, k_rc = 0, None
        while True:
            rc += 1
            k_rc = opening_rate(rc)
            if k_rc < args.k_slow:
                break
        enum.release_cutoff = rc
        enum.k_slow = args.k_slow
        logger.info('Rate-dependent enumeration: k-slow = {}'.format(enum.k_slow))
        logger.info('  - corresponding release-cutoff: {} < L < {}'.format(rc-1, rc))
    else:
        if args.release_cutoff is not None:
            enum.release_cutoff = args.release_cutoff
            logger.info('Rate-independent enumeration: release cutoff L = {}'.format(
                enum.release_cutoff))
            logger.info('  - corresponding k-slow: {}'.format(opening_rate(enum.release_cutoff)))
        else:
            logger.info("Rate-independent enumeration:")
            enum.release_cutoff_1_1 = args.release_cutoff_1_1
            logger.info("  - release cutoff for reaction arity (1,1) = {}".format(
                enum.release_cutoff_1_1))
            enum.release_cutoff_1_2 = args.release_cutoff_1_2
            logger.info("  - release cutoff for reaction arity (1,2) = {}".format(
                enum.release_cutoff_1_2))
    if args.k_fast:
        enum.k_fast = args.k_fast
        logger.info('Rate-dependent enumeration: k-fast = {}'.format(enum.k_fast))

    # DEBUGGING
    enum.DFS = not args.bfs
    enum.interactive = args.interactive
    enum.interruptible = args.interruptible

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Run reaction enumeration (or not) #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    enum_start_time = time.time()

    logger.info("")
    if args.profile:
        try:
            import statprof
        except ImportError:
            logger.warning("Python-module statprof not found (pip install statprof-smarkets). Peppercorn profiling disabled.")
            args.profile = False

    if args.dry_run:
        logger.info("Dry run (not enumerating any reactions)... ")
        enum.dry_run()
        logger.info("Done.")
    else:
        logger.info("Enumerating reactions...")
        if args.interactive:
            logger.info("Interactive mode enabled: Fast and slow reactions " + \
                        "will be printed for each complex as enumerated." + \
                        "Press ^C at any time to terminate and write accumulated" + \
                        "complexes to output.")
        if args.profile:
            statprof.start()
            try:
                enum.enumerate()
            finally:
                statprof.stop()
                statprof.display()
        else:
            try:
                enum.enumerate()
            except KeyboardInterrupt:
                logger.warning("Enumeration interrupted by user.")
                # TODO: try to salvage partial results
                # NOTE: what about the --interruptible flag? only for condensation of incomplete?
                return enum, time.time() - enum_start_time, None

        logger.info("Done.")

    enum_end_time = time.time()
    ncomp = sum(1 for _ in enum.complexes)
    nreac = sum(1 for _ in enum.reactions)
    logger.info(f"Enumerated {nreac} reactions ({ncomp} complexes)")

    reac_types = {r.rtype for r in enum.reactions}
    unexpected_reac_types = reac_types - {"branch-3way", "bind21", "open"}
    if unexpected_reac_types:
        logger.warning(f"Unexpected reaction types encountered during enumeration: {unexpected_reac_types}")
    # ~~~~~~~~~~~~~~~~~~~ #
    # Handle condensation #
    # ~~~~~~~~~~~~~~~~~~~ #
    condensed = args.condensed
    detailed = (not args.condensed or args.detailed)
    if condensed:
        logger.info("Output will be condensed to remove transient complexes.")
        if args.profile:
            statprof.start()
            try:
                enum.condense()
            finally:
                statprof.stop()
                statprof.display()
        else:
            enum.condense()
    condensation_end_time = time.time()

    if args.sbml:
        if detailed and condensed:
            logger.error("SBML output can be detailed OR condensed, not both.")
        enum.to_sbml(args.sbml, condensed = condensed)

    output = enum.to_pil(args.output_filename, 
                         detailed=detailed, condensed=condensed, 
                         molarity=args.concentration_unit, time = args.time_unit)

    if output is not None:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write(output)

        logger.info(f"Output saved to {output_filename}")
    else:
        logger.warning("No output was generated.")

    enum_time = enum_end_time - enum_start_time
    condensation_time = condensation_end_time - enum_end_time if condensed else None
    
    return enum, enum_time, condensation_time


def write_output_with_metadata(output_filename: str, metadata: dict, enum_time: float, condensation_time: Optional[float]):
    """Write output file with metadata prepended as JSON comment.
    
    Args:
        output_filename (str): Path to the output file to modify
        metadata (dict): Original metadata from input file
        enum_time (float): Time taken for enumeration in seconds
        condensation_time (float): Time taken for condensation in seconds
    """
    # Read existing output
    with open(output_filename, 'r') as f:
        content = f.read()
    
    # Create enhanced metadata
    enhanced_metadata = metadata.copy()
    enhanced_metadata["enumeration_time_seconds"] = round(enum_time, 3)

    if condensation_time is not None:
        enhanced_metadata.update({
            'condensation_time_seconds': round(condensation_time, 3),
            'total_time_seconds': round(enum_time + condensation_time, 3)
        })
    
    # Write with metadata prepended
    with open(output_filename, 'w') as f:
        f.write(f"# {json.dumps(enhanced_metadata)}\n")
        f.write(content)


def pil_domain_lengths(pil_file: str, is_filename: bool) -> dict[str, int]:
    pildata = dsdio.read_pil(pil_file, is_file=is_filename)
    doms = pildata["domains"]
    #dsdio.clear_io_objects()
    return {d.name: d.length for d in doms.values()} 



def get_cooperative_toehold_length(metadata: dict, domain_lengths: dict) -> Optional[int]:
    """Calculate total cooperative toehold length from metadata and domain lengths"""
    if 'cooperative_toehold' not in metadata:
        return None
    
    coop_domains = metadata['cooperative_toehold']
    return sum(domain_lengths[dom] for dom in coop_domains)




def check_seesaw_viability(ts_params: RateDependentTimescaleSep, params: dict[str], # min_RD_len: int, max_RD_len: int, max_TH_len: int,
                           basepair_energy: float = -1.7) -> None:
    """Check if the timescale separation is viable for seesaw reactions
    
    Args:
        ts_sep (TimescaleSeparation): timescale separation parameters
        min_RD_len (int): minimum length of recognition domain
        max_RD_len (int): maximum length of recognition domain
        max_TH_len (int): maximum length of toehold
    """
    try:
        min_RD_len = min(params["recognition_domain_lengths"].values())
        max_th = params["max_toehold_length"]

        if not (max_th < min_RD_len):
            raise ValueError(f"maximum toehold ({max_th}) must be shorter than minimum recognition domain ({min_RD_len})"
                             " to allow sensible timescale separation (avoiding irreversible occlusion...)")
            # NOTE: could actually still avoid leak reactions in certain cases?

        #if ts_params.enables_reaction(opening_rate(min_RD_len, dG_bp=basepair_energy)):
        if ts_params.enables_reaction(ReactionType.OPENING, min_RD_len, dG_bp=basepair_energy):
            raise ValueError(f"{ts_params} enables recognition domain opening (of length {min_RD_len})")
            # NOTE: may not be a big issue in cases where there are neighboring domains (that stay hybridized)
        
        #if ts_params.disables_reaction(opening_rate(max_th, dG_bp=basepair_energy)):
        if ts_params.disables_reaction(ReactionType.OPENING, max_th, dG_bp=basepair_energy):
            raise ValueError(f"{ts_params} disables toehold opening (of length {max_th})")

        max_migration_seq = max(params["migration_sequence_lengths"])        
        #if ts_params.disables_reaction(canonical_direct_3way_migration_rate(max_migration_seq)):
        if ts_params.disables_reaction(ReactionType.MIGRATION_3WAY, max_migration_seq):
            raise ValueError(f"{ts_params} disables branch migration over sequence (of length {max_migration_seq})")
        
    except KeyError as e:
        raise KeyError(f"Missing metadata item : {e}")


if __name__ == "__main__":
    handler = cfg_logging_handler()

    parser = argparse.ArgumentParser()

    add_peppercorn_args(parser)
    # override default values for k_slow and k_fast as None (0 is awkward, has strong implications here)
    for action in parser._actions:
        if action.dest in ["k_slow", "k_fast"]:
            action.default = None

    parser.add_argument("--skip-timescale-check", action="store_true", help="skip assertions for correct seesaw circuit modeling")
    #parser.add_argument("--visual-crn", action="store_true", help="also print out Visual CRN code")

    args = parser.parse_args()
    match args.verbose:
        case 0: logger.setLevel(logging.WARNING)
        case 1: logger.setLevel(logging.INFO)
        case _: logger.setLevel(logging.DEBUG)


    if args.input_filename is None:
        raise NotImplementedError("read from stdin")
    
    if args.max_complex_size < 4:
        logger.warning("max complex size < 4 disables pairwise annihilation")
    
    dsdio.set_io_objects()
    dom_lens = pil_domain_lengths(args.input_filename, is_filename=True)
    metadata = read_metadata(args.input_filename, is_filename=True)

    cooperative_th_len = get_cooperative_toehold_length(metadata, dom_lens)
    if cooperative_th_len is None:
        logger.info("no cooperative toehold specified in metadata, using default timescale separation")
        if args.release_cutoff is None:
            args.release_cutoff = 7
            logger.warning(f"no release cutoff specified, defaulting to {args.release_cutoff}")

        timescale_sep = RateIndependentTimescaleSep(release_cutoff=args.release_cutoff)
        if args.k_slow is not None or args.k_fast is not None:
            raise NotImplementedError("allow setting k_slow or k_fast without cooperative hybridization?")
                # still do the assertions
    else:
        # overwrite timescale separation params 
        if cooperative_th_len > metadata["max_toehold_length"]:
            logger.error(f"cooperative toehold length ({cooperative_th_len}) exceeds stored maximum toehold length ({metadata['max_toehold_length']})")

        anh_timescale = cooperative_hybridization_limits(cooperative_th_len, args, tight_lower_bound=True)
        k_slow_max, k_fast_min = anh_timescale.k_slow, anh_timescale.k_fast

        if args.k_slow is None:
            args.k_slow = k_slow_max
        elif args.k_slow > k_slow_max:
            logger.warning(f"overwriting k_slow")
            args.k_slow = k_slow_max

        if args.k_fast is None:
            args.k_fast = k_fast_min
        elif args.k_fast < k_fast_min:
            logger.warning(f"overwriting k_fast")
            args.k_fast = k_fast_min

        timescale_sep = RateDependentTimescaleSep(args.k_slow, args.k_fast)


    # make sure no intended reactions negligible
    if not args.skip_timescale_check:
        check_seesaw_viability(timescale_sep, metadata)
    
    # enumerate
    suffix = "ENUM-CRN"
    if args.condensed: suffix += "cond"
    ppcorn_out_path = os.path.splitext(args.input_filename)[0] + f"_{suffix}_vmax{args.max_complex_size}.pil"
    
    start_time = time.time()
    try:
        enum, enum_time, condensation_time = run_peppercorn(args, ppcorn_out_path)
    except PolymerizationError as e:
        logger.error(f"Peppercorn enumeration failed: {e}")
        sys.exit(1)

    end_time = time.time()
    total_time = end_time - start_time

    # Write metadata to output file
    metadata["product_max_strands"] = args.max_complex_size
    write_output_with_metadata(ppcorn_out_path, metadata, enum_time, condensation_time)

    logger.info(f"Enumeration {'and condensation ' if args.condensed else ''}completed in {total_time:.2f} seconds")

    dsdio.set_io_objects() # should we really do this again? (TODO: move reading and setup to own function)
    print_CRN_stats(dsdio.read_pil(ppcorn_out_path, is_file=True), args.condensed)

    # if args.visual_crn: write_vdsd(enum, detailed=True, condensed=False) # TODO: allow condensed