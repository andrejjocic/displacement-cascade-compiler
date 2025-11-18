# displacement-cascade-compiler
compile and verify DNA strand displacement cascades with seesaw gates and cooperative hybridization

- run `src/compile.py` to apply classifier translation scheme (yielding initial DSD species)
- run `src/enumerate.py` to enumerate reactions and estimate rate constants
- run `src/simulate.py` to compile CRN to ODE system and simulate