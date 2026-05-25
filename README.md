# DNA strand displacement cascade compiler

This compiler enables *in silico* evaluation and verification of DNA strand displacement cascades. The pipeline is similar to [Nuskell](https://github.com/DNA-and-Natural-Algorithms-Group/nuskell)[^5], but specialized for [seesaw circuits](https://qianlab.caltech.edu/SeesawCompiler/) extended with cooperative hybridization[^1].

## Setup

```bash
pip install -r requirements.txt
```
## Usage

### Translation Scheme Implementation

First, you must choose the signal-processing operations you wish to cascade together into a "circuit". 
From these, a domain-level strand displacement (DSD) system is assembled following a "translation scheme"[^5].
The script [src/compile.py](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/src/compile.py) can assemble the DSD system implementing a winner-take-all[^1] (WTA)  or loser-take-all[^2] (LTA) pattern classification circuit given a weight matrix. Run `python src/compile.py -h` for more information.

You can also assemble a custom cascade of supported operations (see table below). 
Unlike Nuskell, we don't provide a high-level language for expressing translation schemes. Instead, you must programmatically string together `CircuitModule` instances into a `DSDCircuit` (defined in [compile.py](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/src/compile.py)). The below snippet shows how to assemble a WTA activation function[^1].


```Python
circ = DSDCircuit()
competitor = circ.new_signal(domain_prefix="c", dimension=N)
bin_out = circ.new_signal(domain_prefix="o", dimension=N)

cascade = [
	PairwiseAnnihilation(input_signal=competitor),
    SignalRestoration(input_signal=competitor, output_signal=binary_response),
    Reporting(input_signal=binary_response)
]
circ.add_modules(cascade, add_reporting=False)
```

It starts by declaring $N$-dimensional input and output `Signal`s. The instance `competitor` is essentially a generator of domains $c_i$ ($0 \leq i < n$). By default, every signal gets its own toehold domain, which reduces the runtime of [enumeration](#reaction-enumeration-and-simulation). You can toggle this at any point with `circ.disable_toehold_generation` and `circ.enable_toehold_generation` if you wish to benchmark the increased effects of toehold occlusion that come with using a global/shared toehold.

Then, the circuit topology is defined as a list of `CircuitModule`s. Aside from the first and last, the order of modules in the list is irrelevant: the data flow topology is defined by connecting modules with `Signal`s. The compiler assumes that the input signal of the *first* module in the list is the actual circuit input (which will be set during [simulation](#reaction-enumeration-and-simulation)). Note that in this example, two modules share that input signal. By default, a `Reporting` module for the output of the *last* module in the list is added to the circuit implicitly (explicit in above example). Reporting is recommended for interpretable results, as output signals may oscillate without it. It also avoids the issue where one formal species is represented by multiple actual species (with differing "history domains"[^5]). Plotting simulation outputs for such cases is currently not implemented.

See [circuit_assembly.ipynb](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/notebooks/circuit_assembly.ipynb) for more examples of cascading operations. 
Currently, only the operations described in papers [^1] and [^2] are implemented. The specification of supporting species (output of translation scheme) is summarized in the table below.

| Module (operation) | Species | Kernel notation | Initial Relative Concentration |
|---|---|---|---|
| weight multiplication; matrix $W \in \mathbb{R}^{n\times m}$ | $W_{ij}$, $i \in [1,n]$, $j \in [1,m]$ | $y_j \text{ } t_y( \text{ } x_i( \text{ } + \text{ } t_x^{\ast} \text{ } ) \text{ } )$ | $w_{ij}$ |
| ↳ multiplication fuel | $XF_{i}$, $i \in [1, n]$ | $t_y \text{ } x_i$ | $\geq \sum_j w_{ij}$ |
| summation (WTA) | $SG_{i}$, $i \in [1,m]$ | $t_x^{\ast} \text{ } x_i^{\ast}( \text{ } t_y^{\ast}( \text{ } + \text{ } y_i \text{ } ) \text{ } )$ | $\geq 1$ |
| simultaneous summation and signal reversal (LTA) | $SRG_{ij}$, $i,j \in [1,m]$, $i \neq j$ | $t_x^{\ast} \text{ } x_i^{\ast}( \text{ } t_y^{\ast}( \text{ } + \text{ } y_j \text{ } ) \text{ } )$ | $\geq \frac{1}{m-1}$ |
| pairwise annihilation | $Anh_{jk}$, $1 \leq i < j \leq m$ | $t_e^{\ast} \text{ } t_x^{\ast} \text{ } x_j^{\ast}( \text{ } x_k( \text{ } + \text{ } t_e^{\ast} \text{ } t_x^{\ast} \text{ } ) \text{ } )$ | $\geq 1$ |
| signal restoration | $RG_i$, $i \in [1,m]$ | $t_x^{\ast} \text{ } x_i^{\ast}( \text{ } t_y^{\ast}( \text{ } + \text{ } y_i \text{ } ) \text{ } )$ | $1$ |
| ↳ restoration fuel | $F_i$, $i \in [1,m]$ | $t_y \text{ } x_i$ | $\geq 1$ |
| reporting | $Rep_{i}$, $i \in [1,m]$ | $t_x^{\ast} \text{ } x_i^{\ast}( \text{ } + \text{ } )$ | $\geq 1$ |

Secondary structures given in [kernel notation](https://github.com/DNA-and-Natural-Algorithms-Group/peppercornenumerator). All species are at most 2-stranded, so their strands can be written in any order[^4].
We use a "parametric kernel notation" where $x_i$ denotes a given module's input signal branch-migration domain, and likewise $y_j$ for the output signal. Domains $t_x$ and $t_y$ are the signals' respective toeholds. For brevity, the toehold extension $t_e$ is listed only in the annihilators, although it's also necessary in any module that directly precedes pairwise annihilation (summation gates $SG$ for WTA, reversal gates $SRG$ for LTA).

Feel free to implement new subclasses of `CircuitModule` to add new operations.
For convenience, you can use the `make_complex` wrapper for constructing a secondary structure from "programmatic kernel notation" instead of the slightly more cumbersome "dot-paren notation" (see implemented examples of `CircuitModule.compile`).
If your translation scheme needs any domains aside from the ones accessible through `Signal`s, you can declare additional global domains using `DSDCircuit.add_global`. 



### Reaction Enumeration and Simulation

Next, we compute all possible reactions in the DSD system.
The script [src/enumerate.py](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/src/enumerate.py) is a wrapper around the [Peppercorn reaction enumerator](https://github.com/DNA-and-Natural-Algorithms-Group/peppercornenumerator) which chooses appropriate default parameter values given the chosen signal processing operations, and performs helpful assertions in case you want to use custom parameter values. Note that proper modeling of `PairwiseAnnihilation` requires non-default timescale separation parameters.
Run `python src/enumerate.py -h` for more information.

After enumeration, run `python src/simulate.py` to simulate the circuit with given input signal values.

### Formal Verification

Alternatively, you can formally verify the correctness of a circuit implementation for *any* input.
The DSD system resulting from a particular execution of a translation scheme may be formally verified using the notion of bisimulation equivalence[^3]. Some helper functions are provided in [verification_helpers.py](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/src/verification_helpers.py).
See [notebook](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/notebooks/formal_verification.ipynb) for examples of usage.




[^1]: [Scaling up molecular pattern recognition with DNA-based winner-take-all neural networks](https://www.nature.com/articles/s41586-018-0289-6) (Cherry and Qian, 2018)

[^2]: [A Loser-Take-All DNA Circuit](https://pubs.acs.org/doi/10.1021/acssynbio.1c00318) (Rodriguez *et al.*, 2021)

[^3]: [Verifying chemical reaction network implementations: A bisimulation approach](https://www.sciencedirect.com/science/article/pii/S0304397518300136) (Johnson *et al.*, 2019)

[^4]: [A domain-level DNA strand displacement reaction enumerator allowing arbitrary non-pseudoknotted secondary structures](https://royalsocietypublishing.org/rsif/article/17/167/20190866/36183/A-domain-level-DNA-strand-displacement-reaction) (Badelt *et al.*, 2020)

[^5]: [A General-Purpose CRN-to-DSD Compiler with Formal Verification, Optimization, and Simulation Capabilities](https://www.springerprofessional.de/en/a-general-purpose-crn-to-dsd-compiler-with-formal-verification-o/14236826) (Badelt *et al.*, 2017)