import json
import pandas as pd
from typing import List



def get_mcycle_section_idx(iteration):
    return 1 + (iteration-1) * 2

def read_rtl_perf_trace(perf_folder, clusters_to_ignore:List, iterations) -> pd.DataFrame:
    dm_cores_hartid = [9*id + 8 for id in range(16)]
    dm_cores_hartid_hex = [hex(id)[2:] for id in dm_cores_hartid]
    perf_files = [f'{perf_folder}/hart_{id.zfill(5)}_perf.json' for id in dm_cores_hartid_hex]


    row_dicts = []
    for i, perf_file in enumerate(perf_files):

        with open(perf_file) as json_file:
            if i in clusters_to_ignore:
                row_dicts.append(dict(cluster_id=dm_cores_hartid[i], cycles=0))
            else:
                perf_data = json.load(json_file)
                cycles = 0
                for iteration in iterations:
                    cycles += perf_data[get_mcycle_section_idx(iteration)]['cycles']
                cycles=round(cycles/len(iterations))
                row_dicts.append(dict(cluster_id=dm_cores_hartid[i], cycles=cycles))

    df = pd.DataFrame(row_dicts)

    return df


def get_cycles_dmcpyi_to_mcycle(log_folder, clusters_to_ignore:List, iteration) -> pd.DataFrame:
    dm_cores_hartid = [9*id + 8 for id in range(16)]
    dm_cores_hartid_hex = [hex(id)[2:] for id in dm_cores_hartid]
    trace_files = [f'{log_folder}/trace_hart_{id.zfill(5)}.txt' for id in dm_cores_hartid_hex]
    cycles_row_dicts = []
    for i, trace_file in enumerate(trace_files):
        if i in clusters_to_ignore:
            cycles_row_dicts.append(dict(cluster_id=dm_cores_hartid[i], cycles=0))
            continue
        row_dicts = []
        with open(trace_file) as tf:
            lines = tf.readlines()
            for line in lines:
                if line[0] == '#':
                    break
                split_line = line.split()
                if len(split_line) == 0:
                    continue
                if split_line[0] == 'M':
                    continue

                row_dicts.append(dict(timestamp=int(split_line[0]), cycle=int(split_line[1]), instruction=split_line[4]))

        df = pd.DataFrame(row_dicts)

        df_n = df.loc[(df['instruction']=='dmcpyi') | (df['instruction']=='csrr')]
        cycles_row_dicts.append(dict(cluster_id=dm_cores_hartid[i], cycles=df_n.iloc[-4]['cycle'] - df_n.iloc[-5]['cycle']))


    df_cycles = pd.DataFrame(cycles_row_dicts)
    return df_cycles
