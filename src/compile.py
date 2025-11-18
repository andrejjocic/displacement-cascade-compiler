"""
Compile a WTA/LTA classification circuit using translation scheme from [Cherry,Qian] / [Rodriguez+]
"""

import itertools
from typing import Generator, Optional, Any, List, Callable
from abc import ABC, abstractmethod
from enum import Enum
import argparse
import numpy as np
import math
import os
import json
from dataclasses import dataclass
import logging
from pathlib import Path
import logging
import pandas as pd

from dsdobjects import DomainS, ComplexS

# FIXME: why does one work in jupyter and the other in script?
from utils import Strand, strand3to5
#from .utils import Strand, strand3to5

# Configure logging for this module
from utils import cfg_logging_handler, set_handle_verbosity
logger = logging.getLogger(__name__)



@dataclass
class FormalReaction:
    """rate-free irreversible formal reaction."""
    reactants: List[str]
    products: List[str]

    def __str__(self) -> str:
        """.crn format"""
        return " + ".join(self.reactants) + " -> " + " + ".join(self.products)
    
    def list_format(self) -> list[list[str]]:
        """list format suitable for CRN equivalence testing."""
        return [self.reactants, self.products]
    
    def inverse(self) -> 'FormalReaction':
        """Return the inverse reaction."""
        return FormalReaction(reactants=self.products.copy(), products=self.reactants.copy())

    def excluding(self, exclude_condition: Callable[[str], bool]) -> 'FormalReaction':
        """Return a new FormalReaction with all species in species_set removed from reactants and products."""
        new_reactants = [s for s in self.reactants if not exclude_condition(s)]
        new_products = [s for s in self.products if not exclude_condition(s)]
        return FormalReaction(new_reactants, new_products)


@dataclass
class Signal:
    """n-dimensional signal (set of n signal species that play similar role in the circuit)"""
    recognition_domain_prefix: str
    """name prefix for all recognition domains"""
    recognition_domain_length: int
    """length of all recognition domains"""
    dim: int
    """dimension (number of formal species)"""
    toehold: DomainS 
    """toehold whose binding initiates branch migration over the recognition domains
    (in the course of standard signal propagation; excluding fuel binding in catalytic cycles)"""

    # TODO: history domain pattern matching

    def __post_init__(self):
        if self.recognition_domain_length <= 0:
            raise ValueError("Recognition domain length must be positive")
        if self.dim <= 0:
            raise ValueError("Signal dimension must be positive")
        
    def domain(self, i: int) -> DomainS:
        """Return the recognition domain for the i-th signal."""
        if not i in range(self.dim):
            raise IndexError(f"Signal index {i} out of range for {self.recognition_domain_prefix} (0-{self.dim-1})")
        
        return DomainS(f"{self.recognition_domain_prefix}{i}", self.recognition_domain_length)
    

    def formal_name(self, i: int) -> str:
        """name of the i-th formal species in this signal."""
        if not i in range(self.dim):
            raise IndexError(f"Signal index {i} out of range for {self.recognition_domain_prefix} (0-{self.dim-1})")
        
        return f"{self.recognition_domain_prefix.upper()}{i}"
    

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            raise TypeError(f"Cannot compare Signal with {type(other)}")
        
        # only allow one signal with the same domain prefix
        return self.recognition_domain_prefix == other.recognition_domain_prefix 

    def __hash__(self) -> int:
        """Hash based on the recognition domain prefix and length."""
        return hash(self.recognition_domain_prefix)


TH_EXT_NAME = "s"
"""name of the toehold extension domain used in cooperative toehold design"""

class DSDCircuit:
    """Domain-level strand displacement circuit.
    Stores only relative concentrations, absolute is set on export.
    """
    _modules: List['CircuitModule']
    """references to all modules added to the circuit (for bookkeeping)"""
    _supporting_species: set[str]
    """set of names of all supporting species added to the circuit so far"""

    _init_rel_conc: dict[ComplexS, float]  # hash(ComplexS) is hash of its canonical form (why not just use the name, given singleton class?)
    """mapping of complexes to their initial relative concentrations"""

    _domains: set[DomainS] # hash(DomainS) is hash of its name
    """set of all domains in the circuit"""
    fuel_multiplier: float
    """relative excess of all fuels (1 is enough for idealized simulation,
    but in practice we need more d.t. effective concentration uncertainty)"""

    globals: dict[str, DomainS]  
    """all common domains, keyed by domain name"""

    _inputs: Optional[List[ComplexS]]
    _outputs: Optional[List[ComplexS]]

    _metadata: dict[str, Any]

    signals: set[Signal]
    """all signals used in the circuit"""
    generated_toeholds: int
    toehold_generation: bool


    def __init__(self, fuel_multiplier: float = 2.0, new_toehold_generation: bool = True):
        if fuel_multiplier < 1:
            raise ValueError("Fuel multiplier must yield excess (relative >= 1x minimum)")
        self.fuel_multiplier = fuel_multiplier

        self._modules = []
        self._supporting_species = set()

        self._init_rel_conc = {}
        self._domains = set()
        self.globals = {}
        self._inputs = None
        self._outputs = None
        self._metadata = {} 
        self.signals = set()  
        self.generated_toeholds = 0
        self.toehold_generation = new_toehold_generation


    def new_toehold(self, length: int = 5) -> DomainS:
        """Create a new toehold domain with the given length."""
        if length <= 0:
            raise ValueError("Toehold length must be positive")
        
        name = f"T{self.generated_toeholds}"
        self.generated_toeholds += 1
        dom = DomainS(name, length)
        
        if dom in self._domains:
            raise ValueError(f"Toehold domain '{name}' already exists in the circuit")
        
        self._domains.add(dom)
        return dom
    
    def last_toehold_used(self) -> DomainS:
        """Return the most recently created toehold domain. If none exist, create the first one."""
        if self.generated_toeholds == 0:
            return self.new_toehold() # create the first toehold if none exist
        else:
            return DomainS(f"T{self.generated_toeholds - 1}")


    def enable_toehold_generation(self):
        """Enable automatic toehold generation for new signals."""
        self.toehold_generation = True

    def disable_toehold_generation(self):
        """Disable automatic toehold generation for new signals."""
        self.toehold_generation = False


    def new_signal(self, domain_prefix: str, dimension: int,
                   domain_len: int = 15, toehold_len: int = 5) -> Signal:
        """Create a new signal with the given prefix and domain length."""

        th = self.new_toehold(length=toehold_len) if self.toehold_generation else self.last_toehold_used()
        sig = Signal(domain_prefix, domain_len, dimension, toehold=th) # TODO: automatic prefix generation
                      

        if sig in self.signals:
            raise ValueError(f"Signal with prefix '{domain_prefix}' is already used in the circuit")
        
        self.signals.add(sig)
        return sig
    

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the circuit."""
        if key in self._metadata:
            logger.warning(f"Overriding existing metadata for {key}")
        self._metadata[key] = value

    def update_metadata(self, key: str, value: Any):
        """add value to existing collection at metadata key, or create a new key if it doesn't exist."""
        if key in self._metadata:
            self._metadata[key] += value 
        else:
            self._metadata[key] = value


    def set_inputs(self, inputs: List[ComplexS], warn_override: bool = True):
        """Specify the input species of the circuit. (NOT SET VALUES)"""
        if warn_override and self._inputs is not None:
            logger.warning("Overriding existing DSD circ input specification")
        self._inputs = inputs

    @property
    def input_species(self) -> List[ComplexS]:
        """Return the input species of the circuit."""
        if self._inputs is None:
            raise ValueError("Inputs not set") # is ValueError the right exception?
        return self._inputs # TODO: copy to avoid mutation? (note singleton class); same for outputs

    def set_outputs(self, outputs: List[ComplexS]):
        """Set the outputs of the circuit."""
        if self._outputs is not None:
            logger.warning("Overriding existing DSD circ output specification")
        self._outputs = outputs

    @property
    def output_species(self) -> List[ComplexS]:
        """Return the names of the output species."""
        if self._outputs is None:
            raise ValueError("Outputs not set")
        return self._outputs

    def add_global(self, name: str, length: int) -> DomainS:
        """Add a global domain to the circuit (and return it)."""
        # no need to check for duplicates, DomainS is metaclass Singleton
        self.globals[name] = dom = DomainS(name, length)
        return dom

    def set_concentration(self, species: ComplexS, rel_conc: float, warn_zero: bool = True):
        if rel_conc < 0:
            raise ValueError(f"Relative concentration must be non-negative (got {rel_conc})")
        
        if rel_conc == 0 and warn_zero:
            logger.warning(f"Setting concentration of {species.name} to zero")
            # NOTE: enumerator is smart enough to ignore C0=0 species anyway
        
        if species in self._init_rel_conc and rel_conc != self._init_rel_conc[species]:
            logger.warning(f"Overwriting initial concentration of {species.name} with new value {rel_conc}")

        self._init_rel_conc[species] = rel_conc
        self._domains.update(species.domains)

    def saturate_fuel(self, species: ComplexS, required_rel_conc: float):
        """Set the concentration of a fuel species to AT LEAST the minimum required relative concentration."""
        self.set_concentration(species, required_rel_conc * self.fuel_multiplier)

    def give_name(self, species: ComplexS):
        """record existence of named complex without setting an initial concentration"""
        if species.name is None:
            raise ValueError("Complex must have a name")
        
        if species in self._init_rel_conc:
            logger.info(f"Complex {species.name} already has an initial concentration set; not modifying it")
            return
        
        self.set_concentration(species, 0, warn_zero=False) 
     

    def implementation_complexity(self, count_domains: bool = False) -> int:
        """number of unique complexes (or domains) initially present in the circuit"""
        init_complexes = [comp for comp, cctr in self._init_rel_conc.items() if cctr > 0]
        
        if count_domains:
            doms = set()
            for comp in init_complexes:
                doms.update(comp.domains) # NOTE: should we count complementary pairs as one?

            return len(doms)
        else:
            return len(init_complexes)
        

    def compile_module(self, module: 'CircuitModule') -> None:
        """Add module components (gates, fuels) to the circuit."""
        for species, conc, is_fuel in module.compile():
            self._supporting_species.add(species.name)
            if is_fuel:
                self.saturate_fuel(species, conc)
            else:
                self.set_concentration(species, conc)


    def add_modules(self, modules: List['CircuitModule'], add_reporting: bool = True, skip_metadata: bool = False) -> None:
        """Add circuit modules, compile them, and set inputs/outputs.
        First one in the list is treated as the input module, last one is the output."""
        if add_reporting:
            if modules and isinstance(modules[-1], Reporting):
                logger.warning("Last module is already a Reporting layer, skipping additional reporting layer")
            else:
                modules.append(Reporting(input_sig=modules[-1].output))
        
        if not modules:
            raise ValueError("At least one module must be provided")

        self._modules.extend(modules)

        for module in modules:
            module.global_domains = self.globals # TODO: try do avoid this
            #module.compile(self)
            self.compile_module(module)
        
        if modules[-1].reversible:
            logger.warning("Last module is reversible. Outputs may oscillate without downstream consumption!")

        if skip_metadata: return
        # Set inputs and outputs based on first and last modules
        self.set_inputs(modules[0].all_inputs)
        try:
            self.set_outputs(modules[-1].all_outputs)
        except NotImplementedError:
            logger.warning("Output species not implemented for the last module; outputs not set.")

        # add metadata for enumeration
        pa = next((m for m in modules if isinstance(m, PairwiseAnnihilation)), None)
        if pa is not None:
            self.add_metadata("cooperative_toehold", [pa.input.toehold.name, TH_EXT_NAME])
            self.add_metadata("max_toehold_length", pa.input.toehold.length + self.globals[TH_EXT_NAME].length) # export_PIL may override 
        
        self.update_metadata("migration_sequence_lengths", [sum(dom.length for dom in mod.branch_migration_strand(0)) for mod in modules])


    def formal_CRN(self, include_supporting_species: bool, decompose_cycles: bool = False) -> Generator[FormalReaction, None, None]:
        for module in self._modules:
            for fr in module.formal_reactions(decompose_cycles=decompose_cycles):
                if not include_supporting_species:
                    fr = fr.excluding(lambda s: s in self._supporting_species)
                
                yield fr


    def partial_species_interpretation(self) -> dict[str, List[str]]:
        """Return a partial interpretation from implementation species to formal species (as needed for bisimulation)."""
        formal2imps: dict[str, List[str]] = {}

        # input mapping
        fst_module = self._modules[0]
        for i in range(fst_module.input.dim):
            formal2imps[fst_module.input.formal_name(i)] = [fst_module.input_species_label(i)]

        # outputs of all modules (intermediates and final outputs)
        for mod in self._modules:
            if mod.output is None: continue

            for i in range(mod.output.dim):
                formal2imps[mod.output.formal_name(i)] = [sig_spc.name for sig_spc in mod.all_output_signal_species(i)]


        # crnverifier.crn_bisimulation_test requires mapping from implementation species to (singleton) lists of formal species
        interp = {}
        for formal, imps in formal2imps.items():
            for imp in imps:
                interp[imp] = [formal]

        return interp



    def export_PIL(self, standard_conc: float = 100.0, conc_unit: str = "nM", output_file: Optional[str] = None,
                   placeholder_conc: float = math.tau, metadata: dict = {}, skip_metadata: bool = False,
                   name_intermediates: bool = False) -> None:
        """
        Export the circuit to PIL format.
        
        Args:
            standard_conc: The standard concentration (1x) value
            conc_unit: The concentration unit (e.g., "nM")
            output_file: Optional path to output file. If None, prints to stdout
            placeholder_conc: arbitrary nonzero concentration (so reaction enumerator will not ignore species with as-yet unknown initial condition).
                Ideally use something recognisable, so it's obvious if we forgot to override it later.  
            metadata: Optional metadata (aside from I/O species) to include in the header comment JSON string
        """
        if standard_conc <= 0:
            raise ValueError("standard concentration must be positive")

        # I/O setup
        if self._inputs is None:
            logger.warning("No inputs defined for the circuit.")
        else:
            for inp_cpx in self.input_species:
                logger.debug(f"setting input species {inp_cpx.name} to {placeholder_conc} {conc_unit}")
                self.set_concentration(inp_cpx, placeholder_conc)
                # NOTE: when running the simulation, you have to override these values!

        if self._outputs is not None:
            for out_cpx in self._outputs:
                self.give_name(out_cpx) # for easy identification in the plot

        if name_intermediates:
            for mod in self._modules:
                if mod.output is None: continue

                for i in range(mod.output.dim):
                    for out_cpx in mod.all_output_signal_species(i):
                        self.give_name(out_cpx)

        pil_lines = []

        if not skip_metadata:
            # specify input/ouput species for the circuit (for convenient simulation)
            metadata = metadata.copy() # avoid mutating the original
            metadata.update(self._metadata) # merge with existing metadata in the circuit
            metadata["inputs"] = [cpx.name for cpx in self.input_species]
            
            if self._outputs is not None:
                metadata["outputs"] = [cpx.name for cpx in self.output_species]
            else:
                logger.warning("No outputs defined for the circuit; output species metadata will be empty.")

            metadata["recognition_domain_lengths"] = {sig.recognition_domain_prefix: sig.recognition_domain_length for sig in self.signals}
            metadata["standard_conc"] = (standard_conc, conc_unit)

            metadata.setdefault("max_toehold_length", -1)
            # NOTE: metadata["cooperative_toehold"] accounted for in add_modules
            for sig in self.signals:
                if sig.toehold.length > metadata["max_toehold_length"]:
                    metadata["max_toehold_length"] = sig.toehold.length
                    logging.debug(f"Updated max_toehold_length to {sig.toehold.length} (toehold of signal {sig.recognition_domain_prefix})")

            pil_lines.append(f"# {json.dumps(metadata)}") # header comment

        # Define domains
        # for dom in DomainS._instanceNames.values():  # this WVDict is not local to the circuit, so it will accumulate all domains from this script!
        for dom in sorted(self._domains, key=lambda d: d.name):
            if not dom.is_complement: # FIXME: what if we only ever use the complement?
                pil_lines.append(f"length {dom.name}\t= {dom.length}")
        
        # Set initial conditions
        for species, rel_conc in self._init_rel_conc.items():
            pil_lines.append(f"{species.name}\t= {species.kernel_string}\t@initial {rel_conc * standard_conc} {conc_unit}")
        
        # Output the PIL content
        if output_file is not None:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)  
            if out_path.suffix != '.pil':
                logger.info(f"Appending '.pil' to output file name: {out_path.name}.pil")
                out_path = out_path.with_suffix('.pil')

            with open(out_path, 'w') as f:
                for line in pil_lines:
                    f.write(line + '\n')

            logger.info(f"Exported PIL to {out_path}")
        else:
            for line in pil_lines:
                print(line)


def make_complex(name: str, kernel: List[DomainS | str], from3prime: bool = False) -> ComplexS:
    """Wrapper for ComplexS constructor.
    Take kernel-string list like [~th, ~yi, '(', yj, '(', '+', ~th, ')', ')'].
    By default, it's expected each consecutive strand is given from 5' to 3' end.
    Adjacent structure characters may be combined into a single string, e.g. "(+)".
    """
    # NOTE: why not just use read_pil_line? 
    # - would need to initialize the domains of course, but we are already doing that in the circuit

    # Split any strings with length > 1 into individual characters
    simple_kernel = []
    for item in kernel:
        if isinstance(item, str) and len(item) > 1:
            # Split string into individual characters
            simple_kernel.extend(list(item))
        else:
            simple_kernel.append(item)
        
    kernel = simple_kernel
    if from3prime:
        kernel = list(reversed(kernel))

    sequence = []
    structure = []
    domain_stack = []

    for i, item in enumerate(kernel):
        if isinstance(item, DomainS):
            sequence.append(item)

            if i < len(kernel) - 1 and kernel[i + 1] == '(': # first domain of bound pair
                domain_stack.append(item)
            else: # unpaired domain
                structure.append('.')

        elif item == '(': # first domain of bound pair
            structure.append('(')

        elif item == ')': # complementary domain of bound pair
            try:
                sequence.append(~domain_stack.pop())
                structure.append(')')
            except IndexError:
                raise ValueError("Unmatched ')' in kernel string")
            
        elif item == '+':
            structure.append('+')
            sequence.append('+') # ComplexS constructor wants it in both for some reason (aesthetic?)
        else:
            raise ValueError(f"Invalid item in kernel string: {item}")

    return ComplexS(sequence=sequence, structure=structure, name=name)



# TODO: move all modules to a separate file translation_schemes.py
@dataclass
class CircuitModule(ABC):
    """unit of circuit functionality"""

    input: Signal
    output: Optional[Signal]
    gate_conc: Optional[float] # weight multiplication layer has a whole matrix of "gate concentrations" (TODO: array-like?)
    """concentration of the gate species"""    

    global_domains: dict[str, DomainS] = None
    """reference to global domains in the containing circuit"""

    @property
    def seesaw_toeholds(self) -> tuple[DomainS, DomainS]:
        """returns input/output toeholds for seesaw cycle in this module
        - input toehold is the one that initiates branch migration
        - output toehold is the one that releases the output and (optionally) initiates a reverse reaction"""
        if self.output is None:
            raise Exception("output-less module only has input toehold")
        
        return self.input.toehold, self.output.toehold


    def input_species_label(self, i: int) -> str:
        """name of i-th input species"""
        #return f"InputSignal_{self.input.formal_name(i)}" # NOTE: this one is inconsistent with export_PIL
        return self.input.formal_name(i) 

    #@abstractmethod
    def input_species(self, i: int) -> ComplexS:
        """Return i-th input to this layer. These may be the outputs of the previous layer WITHOUT history domains (if any)"""
        return make_complex(self.input_species_label(i), self.input_strand(i))

    @abstractmethod
    def input_strand(self, i) -> Strand:
        # TODO: default implementation (self.branch_migration_strand + self.input.toehold ?); must not be abstract method then
        pass
    

    @abstractmethod
    def compile(self) -> Generator[tuple[ComplexS, float, bool], None, None]:
        """Yield supporting species (gates, fuels), their initial concentrations, and fuel flags for this module.
        If fuel flag is True, concentration will be multiplied by circuit.fuel_multiplier."""
        pass

    @abstractmethod
    def formal_reactions(self, decompose_cycles: bool = False) -> Generator[FormalReaction, None, None]:
        """Yield formal reactions corresponding to this module."""
        pass


    # @abstractmethod # TODO: phase out
    def output_species(self, i: int) -> ComplexS:
        """Return i-th output of this layer, including history domains (if any)"""
        raise NotImplementedError("output_species method not implemented for this module")

    @abstractmethod
    def all_output_signal_species(self, i: int) -> Generator[ComplexS, None, None]:
        """generate all complexes corresponding to the i-th output signal (including history domains, if any)"""
        pass
        # default: assuming 1-to-1 mapping of inputs to outputs and presence of free release toehold 
        #yield strand3to5(self.branch_migration_strand(i) + [self.output.toehold, self.output.domain(i)])

    @property
    @abstractmethod
    def reversible(self) -> bool:
        """Return True if the layer is reversible (requires downstream consumption of output)."""
        pass

    @property
    def irreversible(self) -> bool:
        return not self.reversible

    def branch_migration_strand(self, i: int) -> Strand:
        """Return the sequence of (one or more, assuming max-helix semantics)
        consecutive domains over which branch migration occurs.

        Default: just the i-th recognition domain of the input signal,
        as in the standard (reversible) seesaw cycle.
        """
        return [self.input.domain(i)]

    @property
    def all_inputs(self) -> List[ComplexS]:
        return [self.input_species(i) for i in range(self.input.dim)]
    
    @property
    def all_outputs(self) -> List[ComplexS]:
        return [self.output_species(i) for i in range(self.output.dim)]
    

LTA_PAPER_DOMAIN_LEN = {
    "input_signal": 9, # x
    "reversed_signal": 11, # y
    "restored_signal": 11, # z
    
    "primary_toehold": 5, # T
    "toehold_extension": 2, # s
}

WTA_PAPER_DOMAIN_LEN = { # TODO: copy from supplemental
    "input_bit": 20, # for the MNIST digit demo
    "intermediate_product": 15, # p
    # "weighted_sum": ..., # s
    # "restored_signal": ..., # y
    # "global_toehold": ..., # T
}


class ParsableEnum(Enum): # https://gist.github.com/ptmcg/23ba6e42d51711da44ba1216c53af4ea

    @classmethod
    def argtype(cls, s: str) -> Enum:
        try:
            return cls[s] # TODO: allow arbitrary case? (also must make sure names are unique)
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"{s!r} is not a valid {cls.__name__}")

    def __str__(self):
        return self.name
    

class WeightMultiplication(CircuitModule):
    """define size attribute as number of outputs (colums)"""
    
    out_strand_generator: Callable[[int, CircuitModule], Strand]
    """gen(i, self) returns ssDNA dangle for all weight molecules W_{?,i}"""

    def __init__(self, input_sig: Signal, output_sig: Signal, weight_matrix: np.ndarray,
                  out_strand: Callable[[int, CircuitModule], Strand]): # TODO: just give handle to the next module?
        
        pattern_sz, num_classes = weight_matrix.shape
        if pattern_sz != input_sig.dim:
            raise ValueError("Input signal dimension must match the number of rows in the weight matrix")
        if num_classes != output_sig.dim:
            raise ValueError("Output signal dimension must match the number of columns in the weight matrix")

        if np.any(weight_matrix < 0):
            raise ValueError("weight matrix must be non-negative")
        
        # Check if any column sums to more than 1
        column_sums = weight_matrix.sum(axis=0)
        columns_above_one = np.where(column_sums > 1 + 1e-9)[0] 
        if len(columns_above_one) > 0:
            logger.warning(f"{len(columns_above_one)} columns in weight matrix sum to values above 1.")
            for col in columns_above_one:
                logger.warning(f"  Column {col} sum: {column_sums[col]:.3f}")
        
        super().__init__(input=input_sig, output=output_sig, gate_conc=None)
        self.weight_matrix = weight_matrix # FIXME: normalization (do it before calling constructor?)
        self.out_strand_generator = out_strand  

    @property
    def reversible(self) -> bool: return True
    
    def input_strand(self, i: int) -> Strand:
        return [self.input.domain(i) , self.input.toehold]

    def formal_reactions(self, decompose_cycles: bool = False):
        raise NotImplementedError() # similar to SignalRestoration

    def compile(self):
        ti, to = self.seesaw_toeholds

        for k in range(self.input.dim):
            if (needed_fuel := self.weight_matrix[k, :].sum()) == 0: # see WTA NN seesaw diagram
                continue # don't need the weight molecules either (all zero)
            
            e = self.input.domain(k) 
            fuel = make_complex(f"Fuel_{e.name}", [e, to], from3prime=True)
            yield fuel, needed_fuel, True
            # NOTE: Peppercorn would ignore C0=0 species anyway (but it would inflate implementation complexity WITHOUT check > 0)

            for i in range(self.output.dim):
                if (w_ki := self.weight_matrix[k, i]) == 0: continue 
                dangle: Strand = self.out_strand_generator(i, self) 
                w_mol = make_complex(f"Weight_{k}_{i}", dangle + [to, '(', e, "(+", ~ti, "))"])
                yield w_mol, w_ki, False

    def output_species(self, i: int) -> ComplexS:
        raise NotImplementedError()

    # NOTE: will reporting behave right, given the extra (extended) toehold on the other side?

    def all_output_signal_species(self, i: int) -> Generator[ComplexS, None, None]:
        raise NotImplementedError()


class Summation(CircuitModule):
    # TODO: add option for reversible input design (as in SignalReversal)
    def __init__(self, input_sig: Signal, output_sig: Signal):
        if input_sig.dim != output_sig.dim:
            raise ValueError("Input and output signals must have the same dimension")
        
        super().__init__(input=input_sig, output=output_sig, gate_conc=1)
        # NOTE: surely gate_conc may be anything >= 1, provided concentration overflow
        #    is handled upstream (by weight normalization)? 

    @property
    def reversible(self) -> bool:
        return True # NOTE: assuming weight dangle doesn't have the full cooperative toehold
        # FIXME: actually irreversible with unconditional output toehold extension??

    def input_strand(self, i: int) -> Strand:
        raise NotImplementedError()

    def formal_reactions(self, decompose_cycles: bool = False):
        raise NotImplementedError() 

    def compile(self):
        ti, to = self.seesaw_toeholds
        s = self.global_domains["s"]

        def summation_gate(p: DomainS, sim: DomainS) -> ComplexS:
            kern = [~ti, ~p, '(', ~s, '(', ~to, '(+', sim, ')))']
            return make_complex(f"SummationGate_{p.name}_{sim.name}", kern)

        for i in range(self.input.dim):
            gate = summation_gate(self.input.domain(i), self.output.domain(i))
            yield gate, self.gate_conc, False

    def branch_migration_strand(self, i: int) -> Strand:
        return strand3to5([self.input.domain(i), self.global_domains["s"]]) # NOTE: assuming the weight dangle also has the 's'

    def output_species(self, i: int) -> DomainS:
        raise NotImplementedError()
        
    def all_output_signal_species(self, i: int) -> Generator[ComplexS, None, None]:
        raise NotImplementedError()
    

class SignalReversal(CircuitModule):
    """inversion of amplitudes"""
    def __init__(self, input_sig: Signal, output_sig: Signal, reversible_design: bool, reversal_gate_conc=2.0):
        if input_sig.dim != output_sig.dim:
            raise ValueError("Input and output signals must have the same dimension")
        
        if reversal_gate_conc < 1/(input_sig.dim - 1):
            raise ValueError("Reversal gate concentration must be in excess (relative >= 1/(n-1))")
        
        super().__init__(input=input_sig, output=output_sig, gate_conc=reversal_gate_conc)
        self.reversible_design = reversible_design

    @property
    def reversible(self) -> bool:
        #return self.input_design != InputDesign.SHORT
        return self.reversible_design


    def input_strand(self, i: int) -> Strand:
        ti, to = self.seesaw_toeholds
        s = self.global_domains[TH_EXT_NAME] 
        x = self.input.domain(i)
        kern_from3 = [ti, x, s, to]
        if self.reversible:
            kern_from3.pop() # remove domain from the 5' end

        return strand3to5(kern_from3)

    def gate_name(self, i: int, j: int) -> str:
        return f"ReversalGate_{i}_{j}"

    def formal_reactions(self, decompose_cycles: bool = False):
        for i in range(self.input.dim):
            for j in range(self.output.dim):
                if i != j:
                    yield (r := FormalReaction(
                        reactants=[self.input.formal_name(i), self.gate_name(i, j)],
                        products=[self.output.formal_name(j)]
                    ))
                    if self.reversible: yield r.inverse()

    def compile(self):
        ti, to = self.seesaw_toeholds
        s = self.global_domains[TH_EXT_NAME]

        for i in range(self.input.dim):
            for j in range(self.output.dim):
                if i != j:
                    x, y = self.input.domain(i), self.output.domain(j)
                    rev_gate = [~ti, ~x, '(', ~s, '(', ~to, '(+', y, ')))']
                    yield make_complex(self.gate_name(i, j), rev_gate), self.gate_conc, False


    def output_species(self, i: int) -> ComplexS:
        raise NotImplementedError("figure out the history domains")
        s, t = self.domains("s T")
        return make_complex(f"Y{i}", [self.domain(f"y{i}"), t, s, self.domain(f"x{i}")])
    
    def all_output_signal_species(self, i: int) -> Generator[ComplexS, None, None]:
        # reversal gate looks the same regardless of reversible/irreversible design
        s, t = self.global_domains[TH_EXT_NAME], self.output.toehold

        for j in range(self.input.dim):
            if j != i:
                yield make_complex(f"{self.output.formal_name(i)}_from_{self.input.formal_name(j)}",
                                    strand3to5([self.input.domain(j), s, t, self.output.domain(i)]))

    
    def branch_migration_strand(self, i: int) -> Strand:
        seq = [self.input.domain(i), self.global_domains[TH_EXT_NAME]]
        # if not self.input_design == InputDesign.SHORT:
        #     seq.append(self.domain("T")) # migration disconnects the reactant (irreversible)
        if self.irreversible:
            seq.append(self.output.toehold)

        return seq
    

class PairwiseAnnihilation(CircuitModule):
    def __init__(self, input_sig: Signal, annihilator_conc: float = 2):
        if annihilator_conc < 1:
            raise ValueError("Annihilator concentration must be in excess (relative >= 1x)")
        
        super().__init__(input=input_sig, output=None, gate_conc=annihilator_conc)

    @property
    def reversible(self) -> bool: return False

    def input_strand(self, i: int) -> Strand:
        return [self.input.domain(i), self.input.toehold, self.global_domains["s"]]

    
    def anh_name(self, i: int, j: int) -> str:
        return f"Annihilator_{i}_{j}"
    
    def formal_reactions(self, decompose_cycles: bool = False):
        for i, j in itertools.combinations(range(self.input.dim), 2):
            yield FormalReaction(
                reactants=[self.input.formal_name(i), self.input.formal_name(j), self.anh_name(i, j)],
                products=[]
            )
    
    def compile(self):
        t = self.input.toehold
        ext = self.global_domains["s"]

        for i, j in itertools.combinations(range(self.input.dim), 2):
            yi, yj = self.input.domain(i), self.input.domain(j)
            anh = [~ext, ~t, ~yi, '(', yj, '(', '+', ~ext, ~t, ')', ')']            
            # we don't treat annihilator as standard fuel, since concentration adjustment helps with order control (?)
            yield make_complex(self.anh_name(i, j), anh), self.gate_conc, False

    def output_species(self, i: int) -> ComplexS:
        raise ValueError("Annihilation has no non-waste outputs")
    
    def all_output_signal_species(self, i: int):
        raise ValueError("Annihilation has no non-waste outputs")
    
    @property
    def all_outputs(self) -> List[ComplexS]:
        # technically formal output is null, but convenient to override
        return [self.input_species(i) for i in range(self.input.dim)]


class SignalRestoration(CircuitModule):
    """restores the input signal to predetermined output value"""

    REL_STANDARD_OUTPUT = 1.0
    """input will be restored to this concentration"""

    def __init__(self, input_sig: Signal, output_sig: Signal):
        if input_sig.dim != output_sig.dim:
            raise ValueError("Input and output signals must have the same dimension")
        
        super().__init__(input=input_sig, output=output_sig, gate_conc=SignalRestoration.REL_STANDARD_OUTPUT)

    @property
    def reversible(self) -> bool: return True

    def input_strand(self, i: int) -> Strand:
        return [self.input.domain(i), self.input.toehold]

    def gate_name(self, i: int) -> str:
        return f"RestorationGate_{i}"
    
    def fuel_name(self, i: int) -> str:
        return f"RestorationFuel_{i}"
    
    def formal_reactions(self, decompose_cycles: bool = False):
        for i in range(self.input.dim):
            if not decompose_cycles:
                yield FormalReaction(
                    reactants=[self.input.formal_name(i), self.gate_name(i), self.fuel_name(i)],
                    products=[self.input.formal_name(i), self.output.formal_name(i)]
                )
            else:
                # cycle forward step
                yield FormalReaction(
                    reactants=[self.input.formal_name(i), self.gate_name(i)],
                    products=[self.output.formal_name(i)] # , f"{self.gate_name(i)}_intermediate"] 
                )

                # cycle reverse step
                yield FormalReaction(
                    reactants=[self.fuel_name(i)], # f"{self.gate_name(i)}_intermediate"],
                    products=[self.input.formal_name(i)]
                )


    def compile(self):
        ti, to = self.seesaw_toeholds

        for i in range(self.input.dim):
            y, z  = self.input.domain(i), self.output.domain(i)
            rg = make_complex(self.gate_name(i), [~ti, ~y, '(', ~to, '(', '+', z, '))'])
            yield rg, SignalRestoration.REL_STANDARD_OUTPUT, False

            fuel = make_complex(self.fuel_name(i), [to, y])
            yield fuel, SignalRestoration.REL_STANDARD_OUTPUT, True


    def output_species(self, i: int) -> ComplexS:
        return next(self.all_output_signal_species(i))

    def all_output_signal_species(self, i: int) -> Generator[ComplexS, None, None]:
        yield make_complex(f"{self.output.formal_name(i)}_from_{self.input.formal_name(i)}",
                            strand3to5([self.input.domain(i), self.output.toehold, self.output.domain(i)]))


class Reporting(CircuitModule):
    """F-Q pair reporting of given signal domains."""
    
    def __init__(self, input_sig: Signal, reporter_conc=2.0): 
        if reporter_conc < 1:
            raise ValueError("Reporter concentration must be in excess (relative >= 1x)") # TODO: treat as fuel?

        super().__init__(input=input_sig, output=None, gate_conc=reporter_conc)

    @property
    def reversible(self) -> bool: return False

    def input_strand(self, i: int) -> Strand:
        return [self.input.domain(i), self.input.toehold]


    def gate_name(self, i: int) -> str:
        return f"Reporter_{i}"

    def formal_reactions(self, decompose_cycles: bool = False):
        for i in range(self.input.dim):
            yield FormalReaction(
                reactants=[self.input.formal_name(i), self.gate_name(i)],
                products=[f"Fluor_{i}"]
            )

    def compile(self):
        for i in range(self.input.dim):
            z = self.input.domain(i)
            rep = make_complex(self.gate_name(i), [~self.input.toehold, ~z, '(+)'])
            yield rep, self.gate_conc, False

    def output_species(self, i: int) -> ComplexS:
        # this is not the flourescent waste product, but this avoids the history domain issue
        return make_complex(f"Quencher_{i}", [self.input.domain(i)])
        # NOTE: why don't we just swap the strand modifications?? (pretend ATTO dye is on the top strand of reporter.. )
    #     return make_complex(f"Fluor_{i}", [self.domain(f"z{i}"), "(", self.domain("T"), "(", self.domain(f"y{i}"), "+))"])


    def all_output_signal_species(self, i: int) -> Generator[ComplexS, None, None]:
        yield self.output_species(i) # need to do the same quencher hack, because no access to the Signal containing history domains


    @property
    def all_outputs(self) -> List[ComplexS]:
        # override for the above hack
        return [self.output_species(i) for i in range(self.input.dim)]
    
    
def compile_LTA_circuit( # just pass argparse.Namespace object? (no good for grid search / differentiable tuning)
        num_signals: int, use_global_th: bool = False,
        t_len: int = LTA_PAPER_DOMAIN_LEN["primary_toehold"], s_len: int = LTA_PAPER_DOMAIN_LEN["toehold_extension"],
        anh_rel_conc: float = 4.0, global_fuel_multiplier: float = 2.0
    ) -> DSDCircuit:
    
    circ = DSDCircuit(fuel_multiplier=global_fuel_multiplier, new_toehold_generation=not use_global_th)

    # add toeholds
    if s_len == 0:
        raise NotImplementedError("account for lack of toehold extension in all layers")

    circ.add_global("s", s_len)

    input_sig = circ.new_signal("p", dimension=num_signals, domain_len=LTA_PAPER_DOMAIN_LEN["input_signal"])

    # reversed signal Y_i
    competitor = circ.new_signal("rs", dimension=num_signals, domain_len=LTA_PAPER_DOMAIN_LEN["reversed_signal"])

    # argmax/min output
    binary_response = circ.new_signal("bo", dimension=num_signals, domain_len=LTA_PAPER_DOMAIN_LEN["restored_signal"])

    layers: list[CircuitModule] = [
        SignalReversal(input_sig, competitor, reversible_design=False),
        PairwiseAnnihilation(competitor, anh_rel_conc),
        SignalRestoration(competitor, binary_response)
    ]
    circ.add_modules(layers, add_reporting=True)
    return circ


def load_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [col for col in df.columns if col.startswith('memory')]
    logger.info(f"Loading memories from columns: {cols}")
    if not cols:
        raise ValueError("No columns starting with 'memory' found in the Excel file.")
    memories = df[cols].values
    return memories
    


def compile_network(weights: np.ndarray, fuel_multiplier: float = 2.0,
                    anti_classification: bool = False, no_reversal: bool = False, reversible_sig_rev: bool = False) -> DSDCircuit:
    """
    Compile a neural network layer with the given weights into a DSD circuit.
    
    Args:
        weights (np.ndarray): (n, m) array of weights for m memories having each n values
        anti_classification (bool): If True, activation function is LTA (which class the example is NOT in); else WTA
    """
    wta = not anti_classification
    pattern_sz, n_patterns = weights.shape

    if n_patterns < 3 and not wta:
        logger.info(f"Anti-classification mode is needlessly complicated for {n_patterns} patterns!")

    if anti_classification and no_reversal:
        # merge weight multiplication and signal reversal steps
        n = n_patterns
        new_weights = 1/(n-1) * weights @ (np.ones((n, n)) - np.eye(n))
        return compile_network(new_weights, anti_classification=False, fuel_multiplier=fuel_multiplier)
        # TODO: test me on smaller matrices
        
    net = DSDCircuit(fuel_multiplier=fuel_multiplier, new_toehold_generation=True)
    
    # bits of input example (for classification)
    binary_input = net.new_signal("bi", dimension=pattern_sz, domain_len=WTA_PAPER_DOMAIN_LEN["input_bit"])

    # intermediate product    
    intermed_product = net.new_signal("p", dimension=n_patterns, domain_len=WTA_PAPER_DOMAIN_LEN["intermediate_product"]) # technically LTA_PAPER_DOMAIN_LEN["input_signal"] for LTA?

    #net.disable_toehold_generation()

    # toehold extension for reaction order control in the competition stage
    th_ext = net.add_global("s", LTA_PAPER_DOMAIN_LEN["toehold_extension"]) 

    if wta:
        # (weighted sum) pattern similarity
        competitor = net.new_signal("si", dimension=n_patterns, domain_len=LTA_PAPER_DOMAIN_LEN["input_signal"]) # FIXME: why this length??

        def g(i: int, module: WeightMultiplication) -> Strand:
            # assuming summation layer is next
            return strand3to5([module.output.domain(i), th_ext])
            # TODO: toggle for summation reversibility
        
        weight_output_generator = g
    else:
        # reversed signal Y_i
        competitor = net.new_signal("rs", dimension=n_patterns, domain_len=LTA_PAPER_DOMAIN_LEN["reversed_signal"])
        
        # TODO: why not just extract the strand from next layer's input_species
        def f(i: int, module: WeightMultiplication) -> Strand:
            if not reversible_sig_rev:
                return strand3to5([module.output.domain(i), th_ext, competitor.toehold]) # default append full cooperative toehold
            else:
                return strand3to5([module.output.domain(i), th_ext])
        
        weight_output_generator = f


    # argmax/min output
    binary_response = net.new_signal("bo", dimension=n_patterns, domain_len=LTA_PAPER_DOMAIN_LEN["restored_signal"])
        
    #net.add_metadata("max_toehold_length", max_th := competitor.toehold.length + th_ext.length)

    ### compile layers
    similarity_layer = WeightMultiplication(input_sig=binary_input, output_sig=intermed_product, weight_matrix=weights,
                                            out_strand=weight_output_generator)
    activation_func = [
        PairwiseAnnihilation(input_sig=competitor, annihilator_conc=4), # 4 used in WTA 100bit
        SignalRestoration(input_sig=competitor, output_sig=binary_response),
    ]
    if wta:
        activation_func.insert(0, Summation(input_sig=intermed_product, output_sig=competitor)) 
        # OPT: get rid of this layer (let product = competitor)       
    else:
        # combine product summation with signal reversal
        activation_func.insert(0, SignalReversal(input_sig=intermed_product, output_sig=competitor,
                                                  reversible_design=reversible_sig_rev))
    
    net.add_modules([similarity_layer] + activation_func, add_reporting=True)
    return net


if __name__ == "__main__":
    
    handler = cfg_logging_handler()

    parser = argparse.ArgumentParser(description="Compile network given weight matrix in .npy format or columns of CSV/Excel file")
    parser.add_argument("output_file", type=str, help="Output PIL file to save the compiled network")
    parser.add_argument("--weight-matrix", "-w", type=str, required=False,
                        help="File containing weight matrix with each memory as COLUMN (.npy file, or name of Excel sheet with columns 'memory[i]')")
    parser.add_argument("--data-file", type=str, help="CSV or Excel file to load memories from")
    parser.add_argument("--lta", action="store_true",
                        help="Use LTA activation function (for anti-classification) instead of WTA (classification)")
    parser.add_argument("--no-reversal", action="store_true", help="implement LTA without signal reversal (just matrix transformation + WTA)")
    parser.add_argument("--reversible-sig-rev", action="store_true", help="Use reversible design for signal reversal")
    parser.add_argument( '-v', '--verbose', action='count', default=0,
        help="Print logging output. (-vv increases verbosity.)")
    
    args = parser.parse_args()
    #set_handle_verbosity(handler, args.verbose)
    match args.verbose:
        case 0: logger.setLevel(logging.WARNING)
        case 1: logger.setLevel(logging.INFO)
        case _: logger.setLevel(logging.DEBUG)

    if os.path.exists(args.output_file):
        raise FileExistsError(f"Output file {args.output_file} already exists!")


    if args.data_file:
        if args.data_file.endswith(".csv"):
            df = pd.read_csv(args.data_file)
        elif args.data_file.endswith(".xlsx"):
            df = pd.read_excel(args.data_file, sheet_name=args.weight_matrix)
        else:
            raise ValueError("Unsupported file format for data file. Use .csv or .xlsx")

        memories = load_matrix(df)
    elif args.weight_matrix.endswith(".npy"):
        memories = np.load(args.weight_matrix)
    else:
        raise ValueError("Unsupported file format for weight matrix (only .npy is supported)")

    if memories.ndim != 2:
        raise ValueError("Input matrix must be a 2D array (memories as columns)")
    if np.any(memories < 0):
        raise NotImplementedError("Negative weights are not supported (no dual-rail)")
    
    pattern_size, n_patterns = memories.shape
    logger.info(f"Loaded weight matrix with {n_patterns} columns (patterns) and {pattern_size} rows (pattern size)")

    #plot_2D_memories(memories)
    
    if np.any(memories.sum(axis=0) > 1 + 1e-9): # small concentration overflow is fine
        logger.info("normalizing weight matrix by maximum sum")
        memories = memories / memories.sum(axis=0).max()

    if os.path.exists(args.output_file):
        raise FileExistsError(f"Output file {args.output_file} already exists!")
    

    net = compile_network(memories, anti_classification=args.lta, no_reversal=args.no_reversal, reversible_sig_rev=args.reversible_sig_rev)
    # TODO: auto-set "no_reversal" based on matrix sparsity
    
    # net.export_PIL(output_file=f"artifacts/{pattern_size}x{n_patterns}-{'lta' if args.lta else 'wta'}.pil")
    net.export_PIL(output_file=args.output_file)
    logger.info(f"num. initially present complexes: {net.implementation_complexity()}")

