from loaders import gvsoc_loader, rtl_loader

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import argparse
import logging
import json5
import os
from pathlib import Path
import pandas as pd
import csv
import time

pio.kaleido.scope.mathjax=None # Disable mathjax to avoid weird bug when saving as pdf

PATTERN='NO_LOAD_SINGLE'

DMCPYI_TO_MCYCLE = False # ENable to use the DMCPYI instruction as the first border, instead of the first mcycle. Not recommend and probably broken


CLOCK_FREQ = 1_000_000_000 # [Hz]
MAX_BW = 64*CLOCK_FREQ / 1_000_000_000 # [GB/s]

# GVSOC_TRACE_FILE = '/scratch/sem24h19/logs/gvsoc/logs_insn.ansi'
# RTL_PERF_FOLDER = '../flooccamy/logs'

RTL_LOGS_DIR = Path('/scratch2/sem24h19/measurements/rtl')
GVSOC_LOGS_DIR = Path('/scratch2/sem24h19/measurements/gvsoc')

DATA_COLUMNS = ['simulator', 'cluster_id', 'pattern', 'trans_size', 'iteration', 'n_trans', 'rw', 'cycles']


# Depending on the benchmark, we want to ignore some clusters, because their instruction trace differs.
def get_clusters_to_ignore(benchmark):
    if benchmark == "PATTERN_ALL_FROM_1_NO_LOAD":
        return [0]
    elif benchmark == "PATTERN_ALL_FROM_1_FULL_LOAD":
        return [0]
    elif benchmark == "PATTERN_ALL_FROM_1_FULL_LOAD_NARROW":
        return [0]
    elif benchmark == "PATTERN_ALL_FROM_1_NO_LOAD_NARROW":
        return [0]
    elif benchmark == "PATTERN_X_LOOPS":
        return []
    elif benchmark == "PATTERN_X_LOOPS_SIMPLE":
        return [12, 13, 14, 15]
    elif benchmark == "PATTERN_DEBUG":
        return [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    return []


def combine_text_bw(data1, data2):
    combined_text = np.array([f"{t1:.2f}<br>{t2:.2f}%" for t1, t2 in zip(data1.flatten(), data2.flatten())]).reshape(data1.shape)
    return combined_text

def combine_text(data1, data2):
    combined_text = np.array([f"{t1}<br>{t2:.2f}%" for t1, t2 in zip(data1.flatten(), data2.flatten())]).reshape(data1.shape)
    return combined_text


def plot_heatmap(title, gvsoc_data=None, rtl_data=None, params=None, with_bw=False):
    fig = make_subplots(rows=1+with_bw, cols=3, subplot_titles=("RTL", "GVSoC", "Difference (RTL - GVSOC)"))
    fig.layout.annotations[0].update(y=1.08)
    fig.layout.annotations[1].update(y=1.08)
    fig.layout.annotations[2].update(y=1.08)

    x_indices = [str(i) for i in range(4)]
    y_indices = [str(j) for j in range(4)]

    payload = params["trans_size"]*params["num_trans"]

    if rtl_data is not None:
        fig.add_trace(go.Heatmap(z=rtl_data, x=x_indices, y=y_indices, coloraxis='coloraxis', text=rtl_data, texttemplate="%{text}"), row=1, col=1)
        vfunc = np.vectorize(convert_cycles_to_bw)
        bw_rtl = vfunc(rtl_data, payload_size=payload)
        bw_perc = (bw_rtl / MAX_BW) * 100
        combined_text_rtl = combine_text_bw(bw_rtl, bw_perc)
        if with_bw:
            fig.add_trace(go.Heatmap(z=bw_rtl, x=x_indices, y=y_indices, coloraxis='coloraxis2', text=combined_text_rtl, texttemplate="%{text}"), row=2, col=1)

    if gvsoc_data is not None:
        fig.add_trace(go.Heatmap(z=gvsoc_data, x=x_indices, y=y_indices, coloraxis='coloraxis', text=gvsoc_data, texttemplate="%{text}"), row=1, col=2)
        vfunc = np.vectorize(convert_cycles_to_bw)
        bw_gvsoc = vfunc(gvsoc_data, payload_size=payload)
        bw_perc = (bw_gvsoc / MAX_BW) * 100
        combined_text_gvsoc = combine_text_bw(bw_gvsoc, bw_perc)
        if with_bw:
            fig.add_trace(go.Heatmap(z=bw_gvsoc, x=x_indices, y=y_indices, coloraxis='coloraxis2', text=combined_text_gvsoc, texttemplate="%{text}"), row=2, col=2)

    if gvsoc_data is not None and rtl_data is not None:
        diff_data = rtl_data - gvsoc_data
        diff_data_perc = (diff_data / rtl_data) * 100
        diff_bw = bw_rtl - bw_gvsoc
        diff_bw_perc = (diff_bw / bw_rtl) * 100
        combine_text_diff = combine_text(diff_data, diff_data_perc)
        combine_text_bw_diff = combine_text_bw(diff_bw, diff_bw_perc)
        fig.add_trace(go.Heatmap(z=abs(diff_data), x=x_indices, y=y_indices, coloraxis='coloraxis', text=combine_text_diff, texttemplate="%{text}"), row=1, col=3)
        if with_bw:
            fig.add_trace(go.Heatmap(z=diff_bw, x=x_indices, y=y_indices, coloraxis='coloraxis2', text=combine_text_bw_diff, texttemplate="%{text}"), row=2, col=3)
    
    pattern = params['benchmark'].replace("PATTERN_", "")
    title_text = f"Pattern: {pattern} - {params['direction']} -  #trans: {params['num_trans']} - trans size: {params['trans_size']} B"
    fig.update_layout(title_text="", title_x=0.5)
        
    fig.update_layout(coloraxis=dict(colorscale='Viridis', cmin=0, colorbar=dict(title='Cycles')))
    if with_bw:
        fig.update_layout(coloraxis2=dict(colorscale='Viridis', cmin=0, cmax=MAX_BW ,colorbar=dict(title='Bandwidth [GB/s]', len=0.45, y=0.19)))
    
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(height=325, width=700)
    
    return fig


def convert_cycles_to_bw(cycles, payload_size=0x1):
    if cycles == 0:
        return np.nan
    bw = (payload_size / (cycles / CLOCK_FREQ)) / 1_000_000_000
    return bw


def get_gvsoc_data(gvsoc_logs_folder, clusters_to_ignore, iterations):
    if DMCPYI_TO_MCYCLE:
        gvsoc_df = gvsoc_loader.compute_cycles_dmcpyi_to_mcycles(gvsoc_loader.read_gvsoc_insn_trace(gvsoc_logs_folder / "logs_insn.ansi"), clusters_to_ignore, iterations)
    else:
        gvsoc_df = gvsoc_loader.compute_cycles(gvsoc_loader.read_gvsoc_insn_trace(gvsoc_logs_folder / "logs_insn.ansi"), clusters_to_ignore, iterations)
    z = np.reshape(gvsoc_df['cycles'], (4,4), order='F')
    return z

def get_rtl_data(rtl_logs_folder, clusters_to_ignore, iterations):
    if DMCPYI_TO_MCYCLE:
        rtl_df = rtl_loader.get_cycles_dmcpyi_to_mcycle(rtl_logs_folder, clusters_to_ignore, iterations)
    else:
        rtl_df = rtl_loader.read_rtl_perf_trace(rtl_logs_folder, clusters_to_ignore, iterations)
    z = np.reshape(rtl_df['cycles'], (4,4), order='F')
    return z

def get_sim_time_gvsoc(gvsoc_logs_folder):
    with open(gvsoc_logs_folder / "time.log", "r") as f:
        text = f.read()
        return float(text.split(":")[1].strip())
    
def get_sim_time_rtl(rtl_logs_folder):
    with open(rtl_logs_folder / "../transcript", "r") as f:
        text = f.read()
        lines=text.split("\n")
        time_line=lines[-3]
        elapsed_time_str = time_line.split()[-1]
        # Split the time string by ":"
        hours, minutes, seconds = map(int, elapsed_time_str.split(":"))
        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds


def run_dir(benchmark: str, direction: str,num_iters: int, num_trans: int, trans_size: int, **_kwargs):
    """Return the simulation run directory for the given parameters."""
    return f"{benchmark}_{direction}_i{num_iters}_t{num_trans}_s{trans_size}"

def logs_dir_path(system_path: Path, **kwargs):
    """Return the logs directory for the given parameters."""
    return system_path / run_dir(**kwargs) / "logs"

def analyze_sweep(cfg: dict):
    results = []

    """Run a sweep of measurements."""
    for exp in cfg["experiments"]:
        for direction in exp["direction"]:
                for num_trans in exp["num_trans"]:
                    for trans_size in exp["trans_size"]:
                        benchmark = exp["benchmark"]
                        num_iters = exp["num_iters"]
                        logging.info(
                            "Analyzing measurements for benchmark %s, direction %s, #iters %d, #trans %d, size %d",
                            benchmark, direction, num_iters, num_trans, trans_size)
                        params = {
                            "benchmark": benchmark,
                            "direction": direction,
                            "num_trans": num_trans,
                            "num_iters": num_iters,
                            "trans_size": trans_size
                        }
                        clusters_to_ignore = get_clusters_to_ignore(benchmark)
                        rtl_data = None
                        gvsoc_data = None
                        sim_time_rtl = None
                        sim_time_gvsoc = None
                        iterations = [i for i in range(2, num_iters+1)] # Dont want the first (two) iteration because the cache is not warmed up yet
                        for simulator in ["rtl", "gvsoc"]:
                            match simulator:
                                case "rtl":
                                    logs_dir = logs_dir_path(RTL_LOGS_DIR, **params)
                                    try:
                                        rtl_data = get_rtl_data(logs_dir, clusters_to_ignore, iterations)
                                        sim_time_rtl = get_sim_time_rtl(logs_dir)
                                    except FileNotFoundError:
                                        logging.warning("No RTL data found for %s", logs_dir)
                                        rtl_data = None
                                case "gvsoc":
                                    logs_dir = logs_dir_path(GVSOC_LOGS_DIR, **params)
                                    gvsoc_data = get_gvsoc_data(logs_dir, clusters_to_ignore, iterations)
                                    sim_time_gvsoc = get_sim_time_gvsoc(logs_dir)
                        fig = plot_heatmap(f"{benchmark}_{direction}_i{num_iters}_t{num_trans}_s{trans_size}", gvsoc_data=gvsoc_data, rtl_data=rtl_data, params=params)
                        # fig.show()
                        fig.write_image(f'plots/limit32/{benchmark}_{direction}_i{num_iters}_t{num_trans}_s{trans_size}.pdf')
                        fig.write_image(f'plots/limit32/{benchmark}_{direction}_i{num_iters}_t{num_trans}_s{trans_size}.svg')
                        fig.write_image(f'plots/limit32/{benchmark}_{direction}_i{num_iters}_t{num_trans}_s{trans_size}.png', scale=5)

                        # fig.write_html(f'plots/{PATTERN}_{PAYLOAD_SIZE}.html')
                        # fig.write_image(f'plots/{PATTERN}_{PAYLOAD_SIZE}.png', width=1200, height=800)
                        results.append(
                            {"payload_size":trans_size*num_trans, 
                             "maximum_system_latency_rtl":np.nanmax(rtl_data), 
                             "maximum_system_latency_gvsoc":np.nanmax(gvsoc_data),
                             "minimum_system_latency_rtl":np.nanmin(rtl_data[np.nonzero(rtl_data)]),
                             "minimum_system_latency_gvsoc":np.nanmin(gvsoc_data[np.nonzero(gvsoc_data)]),
                             "average_system_latency_rtl":np.nanmean(rtl_data[np.nonzero(rtl_data)]),
                             "average_system_latency_gvsoc":np.nanmean(gvsoc_data[np.nonzero(gvsoc_data)]),
                             "std_system_latency_rtl":np.nanstd(rtl_data[np.nonzero(rtl_data)]),
                             "std_system_latency_gvsoc":np.nanstd(gvsoc_data[np.nonzero(gvsoc_data)]),
                             "sim_time_rtl":sim_time_rtl,
                             "sim_time_gvsoc":sim_time_gvsoc,
                            })
                            
    keys = results[0].keys()

    with open('results.csv', 'w') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze measurements")
    

    parser.add_argument(
        "-c",
        "--cfg",
        type=Path,
        help="Configuration file",
        required=True
    )

    args = parser.parse_args()
    f = open(args.cfg, "r")
    cfg = json5.load(f)

    # Log to measurements.log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s")
    
    analyze_sweep(cfg)


if __name__ == '__main__':
    main()
