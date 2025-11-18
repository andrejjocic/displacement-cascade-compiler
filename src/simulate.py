import argparse
import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt
from typing import Optional, List
import sys
import os
import logging
import pandas as pd
import json
import time
from utils import read_metadata, cfg_logging_handler

# Configure logging for this module
logger = logging.getLogger(__name__)



def get_species_names(crn_content, simulator_prog: str | List[str], verbose=False):
    """Extract species names from pilsimulator's output."""

    simulator_prog = simulator_prog if isinstance(simulator_prog, list) else [simulator_prog]
    # dry run of pilsimulator to get species names
    get_labels_process = subprocess.run(
        simulator_prog + ['--list-labels'],
        input=crn_content,
        text=True,
        capture_output=True
    )
    
    if verbose:
        logger.debug(f"simulator dry run return code: {get_labels_process.returncode}")
    
    # Parse the output to extract species names
    species_names = []
    output_lines = get_labels_process.stdout.strip().split('\n')
    
    # Skip the header line and the last line
    # for line in output_lines[1:-1]:
    for line in output_lines[1:]:
        # Extract the middle column (species name)
        match = re.match(r'^\s*\d+\s+(\S+)\s+.*$', line)
        if match:
            species_names.append(match.group(1))
        else:
            logger.warning(f"Unable to parse line: {line}")
            
    if verbose:
        logger.debug(f"Extracted {len(species_names)} species: {', '.join(species_names)}")
            
    return species_names


def parse_simulation_data(data_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the simulation output data into time points and trajectories."""
    lines = data_str.strip().split('\n')
    data_points = []
    
    for line in lines:
        # Skip empty lines or non-data lines
        if not line.strip() or not line[0].isdigit():
            continue
        
        # Parse the line of data (time and concentrations)
        values = [float(x) for x in line.strip().split()]
        data_points.append(values)
    
    if not data_points:
        raise ValueError("No data points found in simulation output")
    
    # Convert to numpy array for easier processing
    data_array = np.array(data_points)
    
    # First column is time, remaining columns are species concentrations
    timedata = data_array[:, 0]
    trajectories = data_array[:, 1:]
    
    return timedata, trajectories


def add_simulation_and_plotting_args(parser):
    sim_group = parser.add_argument_group('Simulation and Plotting parameters')
    sim_group.add_argument("--input-signal", "-i", nargs='+', #type=float|str
                      help="List of relative concentrations for the circuit's input species OR name of column in input CSV.")
    sim_group.add_argument("--input-csv", type=str, help="Path to a CSV file containing input vectors (optional).")
    
    sim_group.add_argument("--duration", type=float, default=2, 
                      help="Final time for simulation (given time units). Default is 2 hours.")
    sim_group.add_argument("--duration-unit", choices=["seconds", "hours"], default="hours",
                      help="Unit for the simulation duration: 'seconds' or 'hours' (default: hours)")
    
    sim_group.add_argument("--timepoints", type=int, default=500,
                      help="Number of timepoints to be interpolated from ODE solution")
    sim_group.add_argument("--log-scale", dest="log_scale", action="store_true", 
                      help="Use logarithmic scale for x-axis")
    sim_group.add_argument("--y-max", type=float, help="Maximum y-axis value for the plot")
    sim_group.add_argument("--label-thresh", dest="label_thresh", type=float, default=0.01, 
                      help="Minimum average relative concentration for species to be in the plot legend (default: 0.01)")
    sim_group.add_argument("--plot-outputs-only", action="store_true",
                      help="Plot only the output species (bypasses label threshold)")


def compile_ODEsys(pil_filename: str, out_path: Optional[str] = None):
    with open(pil_filename, 'r') as f:
        crn_content = f.read()

    if out_path is None:
        base = os.path.splitext(pil_filename)[0]
        out_path = f"{base}_odesys.py"

    t0 = time.time()
    ret = subprocess.run(['pilsimulator', '--dry-run', '--force', '-o', out_path], input=crn_content, text=True, capture_output=True)
    duration = time.time() - t0
    if ret.returncode != 0:
        raise RuntimeError(f"Error compiling ODE system: {ret.stderr}")
    
    logger.info(f"ODE system compiled to {out_path} (in {duration:.2f} seconds).")
    # TODO: copy metadata from pil file to out_path, include compilation time
    #   ensure you don't break shebang!
    return out_path


def get_input_vector(args: argparse.Namespace, normalise: bool = False) -> List[float]:
    
    if args.input_csv:
        col = args.input_signal[0] 
        if (narg := len(args.input_signal)) > 1:
            logger.warning(f"Multiple ({narg}) CSV column names provided, using only the first one ({col})")

        df = pd.read_csv(args.input_csv)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV file '{args.input_csv}'")
        
        vec = df[col]
        if vec.dtype not in [np.float64, np.float32, np.int64, np.int32]:
            raise ValueError(f"Column '{col}' in CSV file '{args.input_csv}' must be numeric (float or int), found {vec.dtype}")
        
        if not all(val in [0, 1] for val in vec):
            raise NotImplementedError("CSV input vector must be binary")

        if normalise:
            col_sum = vec.sum()
            logger.info(f"Normalising input vector by its sum ({col_sum})")
            vec /= col_sum

        # TODO: custom scaling of binary inputs (for rate control)
        return vec.tolist()

    else:
        return [float(c) for c in args.input_signal]


def plot_output_signals(pil_path: str, ax: plt.Axes,
                        input_vector: list[float],
                        supporting_concentrations: dict[str, float] = {},
                        odesys_path: str = "",
                        duration: float = 2.5, duration_unit="hours", timepoints=500,
                        species_remap: dict[str, str] = None,
                        label_final: bool = True,
                        plot_kwargs={}):
    """Plot output signals from a CRN defined in a .pil file.
    
    Args:
        pil_path: Path to the .pil file defining the CRN.
        ax: Matplotlib Axes object to plot on.
        input_vector: List of relative concentrations for the circuit's input species.
        supporting_concentrations: Overrides of relative concentrations for supporting species .
        odesys_path: Optional path to precompiled ODE system. If not provided, will compile from pil_path.
        duration: Duration of the simulation (in given time units).
        duration_unit: Unit for the simulation duration: 'seconds' or 'hours' (default: hours).
        timepoints: Number of timepoints to be interpolated from ODE solution.
        plot_kwargs: Additional keyword arguments to pass to the plotting function."""

    if not odesys_path:
        odesys_path = f"{os.path.splitext(pil_path)[0]}_odesys.py"

    if not os.path.exists(odesys_path):
        logger.info(f"Compiling ODE system from {pil_path}...")
        compile_ODEsys(pil_path, out_path=odesys_path)

    with open(pil_path, 'r') as f:
        crn_content = f.read()

    metadata = read_metadata(crn_content, is_filename=False)

    input_names = metadata["inputs"]
    output_names = metadata["outputs"]

    #input_vector = get_input_vector(args, normalise=False)

    SECONDS_PER_HOUR = 3600
    simulation_duration = duration * SECONDS_PER_HOUR if duration_unit == "hours" else duration # CRN rates are given in per (mole per) second

    simulation_args = [f"--t8={simulation_duration}", f"--t-lin={timepoints}"]
    
    # Get all species names from dry run of simulation; OPT: just use read_pil
    species_names = get_species_names(crn_content, simulator_prog=["python", odesys_path])
    simulation_args += ["--nxy", "--labels"] + species_names

    #simulation_args += ["--nxy","--labels"] + output_names + ["--labels-strict"] # ensure predictable order of output
    
    # set input concentrations
    standard_conc, unit = metadata["standard_conc"]
    if unit != "nM":
        raise ValueError(f"Standard concentration unit {unit} not supported. Please use nM.")
    simulation_args.append("--p0")

    species_names_set = set(species_names) 

    for inp, rel_conc in zip(input_names, input_vector):
        if inp not in species_names_set:
            logger.info(f"Input species '{inp}' not found in CRN. Skipping.") # can happen for sparse matrices
            continue
        simulation_args.append(f"{inp}={rel_conc * standard_conc}")

    for sp, rel_conc in supporting_concentrations.items():
        if sp not in species_names_set: 
            logger.warning(f"Supporting species '{sp}' not found in CRN. Skipping.")
            continue
        simulation_args.append(f"{sp}={rel_conc * standard_conc}")

    logger.debug(f"Running simulation with arguments: {simulation_args}")

    t0 = time.time()
    simulator_process = subprocess.run(
        ["python", odesys_path] + simulation_args, 
        input=crn_content,
        text=True,
        capture_output=True
    )
    sim_duration = time.time() - t0
    
    # Check for errors in simulator
    if simulator_process.returncode != 0:
        logger.error(f"Error running simulator: {simulator_process.stderr}")
        exit(1)

    logger.info(f"Simulation completed in {sim_duration:.2f} seconds.")

    try:
        times, trajectories = parse_simulation_data(simulator_process.stdout)
        if duration_unit == "hours":
            times /= SECONDS_PER_HOUR

        plot_simulation_trajectories(ax, times, trajectories, species_names, output_names, 
                                     standard_conc=standard_conc, species_remap=species_remap,
                                     label_final=label_final, **plot_kwargs)
        return trajectory_dataframe(times, trajectories, species_names, duration_unit)
        
    except Exception as e:
        logger.error(f"Error while processing simulation data: {e}")


def plot_simulation_trajectories(ax: plt.Axes, time_points: np.ndarray, trajectories: np.ndarray,
                                 all_names: list[str], output_names: list[str], standard_conc: float,
                                 species_remap: dict[str, str] = None, label_final=True, **plot_kwargs): 

    traj_indices = [all_names.index(name) for name in output_names]
    rel_trajectories = trajectories[:, traj_indices] / standard_conc

    cycle_styles = "linestyle" not in plot_kwargs
    line_styles = ['-', '--', '-.', ':']

    for i, name in enumerate(output_names):
        style = line_styles[i % len(line_styles)] if cycle_styles else plot_kwargs["linestyle"]
        #plot_kwargs.setdefault('linestyle', style)
        plot_kwargs['linestyle'] = style
        if species_remap:
            name = species_remap.get(name, name)

        lab = name
        if label_final:
            lab += f" (na koncu: {rel_trajectories[-1,i]:.2f})"
        ax.plot(time_points, rel_trajectories[:, i], label=lab, **plot_kwargs)


def trajectory_dataframe(time_points: np.ndarray, trajectories: np.ndarray,
                         species_names: list[str], duration_unit: str) -> pd.DataFrame:
    """Convert simulation trajectories to a pandas DataFrame."""
    data = { f"time ({duration_unit})": time_points }
    
    for i, species in enumerate(species_names):
        data[species] = trajectories[:, i]
    
    df = pd.DataFrame(data)
    return df


def save_simulation_data(time_points, trajectories, species_names, metadata: dict, args: argparse.Namespace, output_only=True):
    """
    Save simulation data as a pandas DataFrame with metadata.
    
    Args:
        time_points: Array of time values
        trajectories: 2D array of species concentrations over time
        species_names: List of species names (column labels)
        metadata: dictionary with additional metadata
        time_unit: Unit for time column
    """
    # Create DataFrame with time and species data
    df = trajectory_dataframe(time_points, trajectories, species_names, args.duration_unit)
    if output_only:
        outs = metadata.get("outputs", [])
    df.drop(columns=[col for col in df.columns if col not in outs and col != f"time ({args.duration_unit})"], inplace=True)

    data = {f"time ({args.duration_unit})": time_points}
    for i, species in enumerate(species_names):
        if output_only and species not in metadata.get("outputs", []):
            logger.info(f"Skipping non-output species '{species}' in saved data.")
            continue
        data[species] = trajectories[:, i]
    
    df = pd.DataFrame(data)
    
    if args.crn_file is not None:
        out_name = os.path.splitext(os.path.basename(args.crn_file))[0]
        out_path = os.path.join(os.path.dirname(args.crn_file), f"{out_name}_simulation-data_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    else:
        out_path = f"simulation-data_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"


    
    with open(out_path, 'w') as f:
        f.write(f"# {json.dumps(metadata)}\n")
        df.to_csv(f, index=False)
    
    logger.info(f"Simulation data saved to {out_path}")
    return df


if __name__ == "__main__":
    handler = cfg_logging_handler()

    parser = argparse.ArgumentParser(description="Simulate a CRN and plot the results.")
    
    parser.add_argument("crn_file", type=str, default=None, nargs='?',
                         help="Path to the CRN file (.pil or .crn). Will read from stdin if not provided.")
    parser.add_argument("--input-format", choices=["crn", "pil"], default="pil")
    parser.add_argument( '-v', '--verbose', action='count', default=0,
        help="Print logging output. (-vv increases verbosity.)")
    parser.add_argument("--compile-only", action="store_true", 
                        help="Only compile the ODE system, do not run simulation")
    parser.add_argument("--force-compile", action="store_true",
                        help="Force recompilation of the ODE system even if it exists")
    parser.add_argument("--save-data", action="store_true", 
                        help="Save simulation data to CSV file with metadata")
    add_simulation_and_plotting_args(parser)

    args = parser.parse_args()
    match args.verbose:
            case 0: logger.setLevel(logging.WARNING)
            case 1: logger.setLevel(logging.INFO)
            case _: logger.setLevel(logging.DEBUG)

    if args.crn_file is not None:
        if not args.crn_file.endswith(f".{args.input_format}"):
            raise ValueError(f"File {args.crn_file} does not have the expected .{args.input_format} extension")

    if args.compile_only:
        assert args.crn_file is not None
        output = compile_ODEsys(args.crn_file)
        logger.info(f"ODE system compiled to {output}. Exiting as per --compile-only flag.")
        exit(0)
    
    crn_data = args.crn_file if args.crn_file else sys.stdin.read()
    
    #assert type(args.input_signal) is list and len(args.input_signal) > 0
    if args.input_signal is None or len(args.input_signal) == 0:
        raise ValueError("Input signal must be provided via --input-signal/-i")

    ax = plt.gca()
    plot_output_signals(
        pil_path=args.crn_file,
        ax=ax,
        input_vector=get_input_vector(args, normalise=False),
        duration=args.duration,
        duration_unit=args.duration_unit,
        timepoints=args.timepoints
    )
    plt.show()