# DNA strand displacement cascade compiler

## Setup

```bash
pip install -r requirements.txt
```
## Usage

Run `python src/compile.py` to assemble the DSD system implementing a WTA[^1] or LTA[^2] classification circuit. See [circuit_assembly.ipynb](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/notebooks/circuit_assembly.ipynb) for compiling arbitrary cascades of molecular signal processing operations. Currently, only operations from the above classifiers are implemented (based on seesaw gates and cooperative hybridization).

Run `python src/enumerate.py` to enumerate reactions in the DSD system and  `python src/simulate.py` simulate them.

See [formal_verification.ipynb](https://github.com/andrejjocic/displacement-cascade-compiler/blob/main/notebooks/formal_verification.ipynb) for formally verifying circuit implementations using bisimulation[^3].

[^1]: [Scaling up molecular pattern recognition with DNA-based winner-take-all neural networks](https://www.nature.com/articles/s41586-018-0289-6) (Cherry and Qian, 2018)

[^2]: [A Loser-Take-All DNA Circuit](https://pubs.acs.org/doi/10.1021/acssynbio.1c00318) (Rodriguez *et al.*, 2021)

[^3]: [Verifying chemical reaction network implementations: A bisimulation approach](https://www.sciencedirect.com/science/article/pii/S0304397518300136) (Johnson *et al.*, 2019)
