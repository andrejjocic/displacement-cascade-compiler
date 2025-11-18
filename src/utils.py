from dsdobjects import ComplexS, DomainS
from dsdobjects.objectio import read_pil, set_io_objects

from typing import List, Dict, Any
import itertools
import json
from collections import Counter
import logging


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'  # Reset color
    
    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)
        
        # Add color based on log level
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        
        return log_message


def set_handle_verbosity(h, v):
    if v == 0:
        h.setLevel(logging.WARNING)
    elif v == 1:
        h.setLevel(logging.INFO)
    elif v == 2:
        h.setLevel(logging.DEBUG)
    elif v >= 3:
        h.setLevel(logging.NOTSET)


def cfg_logging_handler():
    """Configure logging for command-line usage with colors"""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True  # Override any existing configuration
    )
    return handler


def read_metadata(pil_content: str, is_filename: bool) -> dict:
    """Extract metadata from PIL file's header comment"""
    if is_filename:
        with open(pil_content, 'r') as f:
            first_line = f.readline().strip()
    else:
        first_line = pil_content.split('\n', 1)[0] # TODO: handle lack of newline

    if not first_line.startswith('#'):
        raise ValueError("No header comment found at the top of the PIL file")

    try:
        return json.loads(first_line.lstrip('#').strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in header comment: {e}")
    

def write_metadata(filename: str, metadata: dict) -> None:
    """Write metadata as a JSON header comment to the top of the file"""
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0)
        f.write(json.dumps(metadata) + '\n' + content)


def print_CRN_stats(pil_data: Dict[str, Any], condensed: bool) -> None:
    species_category = 'macrostates' if condensed else 'complexes'
    species = pil_data[species_category]
    
    print(f"Total {species_category}: {len(species)}")

    if species_category == 'complexes':
        num_strands = Counter(cpx.size for cpx in species.values())
        for size in sorted(num_strands.keys(), reverse=True):
            print(f"\t{num_strands[size]}\twith {size} strands")

    reactions = pil_data[f"{'con' if condensed else 'det'}_reactions"]
    print(f"Total reactions: {len(reactions)}")

    if not condensed:
        reaction_types = Counter(r.rtype for r in reactions)
        for rtype, count in reaction_types.items():
            print(f"\t{count}\t{rtype}")



def nested(pi: dict[tuple[int, int], tuple[int, int]]) -> bool:
    """Check if the given secondary structure satisfies nesting condition for non-PK"""
    structure_pairs = itertools.combinations(pi.items(), 2)

    for (ij, kl), (pq, uv) in structure_pairs:
        if ij < pq < kl:
            if not (ij < uv < kl): return False # nesting implication not true

    return True


Strand = list[DomainS] # dsdobjects.StrandS seems unfinished? (not meant to be used)
"""list of domains from 5' to 3' end of strand"""

def strand3to5(domains3to5: List[DomainS]) -> Strand:
    """construct a Strand from a list of domains in 3' to 5' order (the "wrong" order)"""
    return list(reversed(domains3to5))

# TODO: proper Strand class with methods push3, push5, pop3, pop5
# - constructor takes bool arg from_3prime
# - need a new name for "backwards compatibility" with existing usage?


def circular_permutations(iterable):
    """Arrangements of objects in a circle. Orders that are cyclic permutations 
    (circular shifts) of each other are considered equivalent and only one is yielded."""

    fst = next(iterable) # arbitrary first element

    for other_perms in itertools.permutations(iterable):
        yield (fst,) + other_perms



class SecondaryStructure:
    strands: dict[str, Strand]
    """strand id -> strand"""
    bindings: dict[tuple[str, int], tuple[str, int]] 
    """(strand id, domain idx) -> other (strand id, domain idx)"""
    # NOTE: can't key by domain (one strand can have multiple instances of the same domain)

    def __init__(self, strands: dict[str, Strand]):
        self.strands = strands
        self.bindings = {}

    @property
    def pseudoknotted(self) -> bool:
        # the nesting condition is invariant to circular shift of strands
        for strand_order in circular_permutations(iter(self.strands.keys())):
            strand_idx = {s: i for i, s in enumerate(strand_order)}
            pi = {(strand_idx[si], j): (strand_idx[sk], l) for (si, j), (sk, l) in self.bindings.items()}
            # OPT: pass only uni-directional mapping to nested()
                # NOTE: *can* we do that without affecting result? 
                # maybe we would just need to perform 2 checks in loop body, defeating purpose
            if nested(pi):
                return False # self is non-pseudoknotted
        
        return True # self has no nested strand order            

    def hybridize(self, strandA: str, domainA: int, strandB: str, domainB: int) -> None:
        if domainA < 0: domainA += len(self.strands[strandA])
        if domainB < 0: domainB += len(self.strands[strandB])

        if not 0 <= domainA < len(self.strands[strandA]):
            raise ValueError(f"Invalid domain index {domainA} for strand {strandA}.")
        if not 0 <= domainB < len(self.strands[strandB]):
            raise ValueError(f"Invalid domain index {domainB} for strand {strandB}.")
        # TODO: just catch IndexError, no need for amending negative indices
        
        if self.strands[strandA][domainA] != ~self.strands[strandB][domainB]:
            raise ValueError(f"Cannot hybridize {strandA}[{domainA}] and {strandB}[{domainB}] because they are not complementary.")
        
        domA = (strandA, domainA)
        domB = (strandB, domainB)
        if domA in self.bindings or domB in self.bindings:
            raise ValueError(f"Cannot hybridize {domA} and {domB} because one of them is already bound.")
        
        self.bindings[domA] = domB
        self.bindings[domB] = domA
        

    @staticmethod
    def from_kernel_notation(kern: str) -> 'SecondaryStructure':
        raise NotImplementedError("TODO")
