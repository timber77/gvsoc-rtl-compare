import logging
from pathlib import Path
import subprocess
import argparse
import json5
import os
import time

FLOOCCAMY_DIR = Path("/home/sem24h19/flooccamy")
GVSOC_DIR = Path("/home/sem24h19/example_floonoc_gvsoc")

# RTL_LOGS_DIR = Path("/scratch/sem24h19/measurements/rtl")
RTL_LOGS_DIR = Path("/scratch2/sem24h19/measurements/rtl")
GVSOC_LOGS_DIR = Path("/scratch2/sem24h19/measurements/gvsoc")


BINARY = FLOOCCAMY_DIR / "sw/build/rtl/intercluster_bench.elf"
#BINARY = FLOOCCAMY_DIR / "sw/build/rtl/sanity.elf"
CFG_H = FLOOCCAMY_DIR / "sw/src/config.h"



def write_cfg_h(cfg_file: Path, params: dict):
    """Write the configuration to the flooccamy config.h file"""
    logging.info("Writing flooccamy configuration to %s", cfg_file)
    with open(cfg_file, "w", encoding="utf-8") as f:
        for k, v in params.items():
            if isinstance(v, int):
                f.write(f"#define {k.upper()} {v}\n")
            else:
                f.write(f"#define {v.upper()}\n")

def build_sw():
    """Build the flooccamy software"""
    logging.info("Building flooccamy software")
    subprocess.run(f"make -C {FLOOCCAMY_DIR} sw", shell=True, check=True)


def run_dir(benchmark: str, direction: str,num_iters: int, num_trans: int, trans_size: int, **_kwargs):
    """Return the simulation run directory for the given parameters."""
    return f"{benchmark}_{direction}_i{num_iters}_t{num_trans}_s{trans_size}"

def logs_dir_path(system_path: Path, **kwargs):
    """Return the logs directory for the given parameters."""
    return system_path / run_dir(**kwargs) / "logs"


def rtl_run_sim(logs_dir: Path):
    """Run the rtl simulation."""
    logging.info("Running rtl simulation in %s", logs_dir)
    subprocess.run(
        f"mkdir -p {logs_dir} && make -C {FLOOCCAMY_DIR} LOGS_DIR={logs_dir} APP={BINARY} run-sim-batch",
        shell=True,
        check=True,
        # env=set_rtl_env(),
    )


def gvsoc_run_sim(logs_dir: Path):
    """Run the gvsoc simulation."""
    logging.info("Running gvsoc simulation in %s", logs_dir)
    start_time = time.time()
    subprocess.run(
        f"mkdir -p {logs_dir} && cd {logs_dir.parent} && make -C {GVSOC_DIR} LOGDIR={logs_dir} WORKDIR={logs_dir} BINARY={BINARY} run", # TODO Check
        shell=True,
        check=True,
        # env=set_gvsoc_env(),
    )
    end_time = time.time()
    with open(logs_dir / "time.log", "w") as f:
        f.write(f"Simulation time in seconds: {end_time - start_time}")
        print(f"Simulation time in seconds: {end_time - start_time}")


def rtl_gen_traces(logs_dir: Path):
    """Generate rtl traces."""
    logging.info("Generating rtl traces in %s", logs_dir)
    subprocess.run(
        f"make -C {FLOOCCAMY_DIR} LOGS_DIR={logs_dir} traces -j",
        shell=True,
        check=True,
    )


def clean_up(logs_dir: Path):
    """Clean up rtl traces and logs."""
    logging.info("Cleaning up rtl traces and logs in %s", logs_dir)
    for file in os.listdir(logs_dir):
        if file.endswith(".dasm"):
            os.remove(logs_dir / file)
    for file in os.listdir(logs_dir.parent):
        if file.endswith(".log") or file.endswith(".tdb"):
            os.remove(logs_dir.parent / file)


def run_sweep(cfg: dict, simulators: list):
    """Run a sweep of measurements."""
    for exp in cfg["experiments"]:
        for direction in exp["direction"]:
            for num_trans in exp["num_trans"]:
                for trans_size in exp["trans_size"]:
                    num_iters = exp["num_iters"]
                    logging.info(
                        "Running measurements for benchmark %s, direction %s, #iters %d, #trans %d, size %d",
                        exp["benchmark"], direction, num_iters, num_trans, trans_size)

                    params = {
                        "benchmark": exp["benchmark"],
                        "direction": direction,
                        "num_trans": num_trans,
                        "num_iters": num_iters,
                        "trans_size": trans_size
                    }

                    # Build the SW
                    write_cfg_h(CFG_H, params)
                    build_sw()  

                    for simulator in simulators:
                        match simulator:
                            case "rtl":
                                logs_dir = logs_dir_path(RTL_LOGS_DIR, **params)
                                rtl_run_sim(logs_dir)
                                rtl_gen_traces(logs_dir)
                                clean_up(logs_dir)
                            case "gvsoc":
                                logs_dir = logs_dir_path(GVSOC_LOGS_DIR, **params)
                                gvsoc_run_sim(logs_dir)




def main():
    """Run the measurements."""

    parser = argparse.ArgumentParser(description="Run measurements")
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        help="Output directory",
        default=Path.cwd().parent / "measurements",
    )
    parser.add_argument(
        "-c",
        "--cfg",
        type=Path,
        help="Configuration file",
        required=True
    )
    parser.add_argument(
        "-s",
        "--simulator",
        type=str,
        action="append",
        help="Simulator to run",
        choices=["rtl", "gvsoc"],
        required=True,
    )

    args = parser.parse_args()
    f = open(args.cfg, "r")
    cfg = json5.load(f)

    # Log to measurements.log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s")
    
    run_sweep(cfg, args.simulator)


if __name__ == "__main__":
    main()
