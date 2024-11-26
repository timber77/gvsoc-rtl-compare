import pandas as pd


def get_mcycle_section_idx(iteration):
    return (iteration-1) * 2

def read_gvsoc_insn_trace(trace_file) -> pd.DataFrame:
    row_dicts = []
    with open(trace_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.split()
            # print(split_line)
            first_slash = split_line[2].index('/')
            row_dicts.append(dict(timestamp=int(split_line[0][:-1]), cycle=int(split_line[1][:-1]), path=split_line[2][first_slash:], function=split_line[4], instruction=split_line[7]))
    df = pd.DataFrame(row_dicts)
    return df

def compute_cycles(df, clusters_to_ignore, iteration):
    row_dicts = []
    df = df.loc[df['function'] == 'snrt_mcycle:21']
    for i in range(16):
        if i in clusters_to_ignore:
            row_dicts.append(dict(cluster_id=i, cycles=0))
        else:
            df_cluster = df[df.path.str.contains(f'cluster_{i}/')]
            i = get_mcycle_section_idx(iteration)
            cycles = (df_cluster.iloc[i+1]['cycle'] - df_cluster.cycle.iloc[i])
            row_dicts.append(dict(cluster_id=i, cycles=cycles))
    return pd.DataFrame(row_dicts)

def cycles_dmcpyi_to_mcycles(df, clusters_to_ignore, iteration):
    print("gvsoc")
    for i in range(16):
        if i in clusters_to_ignore:
            pass
        else:
            df_cluster = df[df.path.str.contains(f'cluster_{i}/')]
            df_n = df_cluster.loc[(df['instruction']=='dmcpyi') | (df_cluster['function']=='snrt_mcycle:21')].tail(3)
            print(f"Cluster {i}: mcycle to dmcpyi: {df_n.iloc[-2]['cycle']-df_n.iloc[-3]['cycle']}   dmcpyi to mcycle: {df_n.iloc[-1]['cycle'] - df_n.iloc[-2]['cycle']}")
    
def compute_cycles_dmcpyi_to_mcycles(df, clusters_to_ignore, iteration):
    row_dicts = []
    for i in range(16):
        if i in clusters_to_ignore:
            row_dicts.append(dict(cluster_id=i, cycles=0))
        else:
            df_cluster = df[df.path.str.contains(f'cluster_{i}/')]
            df_n = df_cluster.loc[(df['instruction']=='dmcpyi') | (df_cluster['function']=='snrt_mcycle:21')].tail(3)
            cycles = (df_n.iloc[-1]['cycle'] - df_n.iloc[-2]['cycle'])
            row_dicts.append(dict(cluster_id=i, cycles=cycles))
    return pd.DataFrame(row_dicts)

def get_structured_data(gvsoc_logs_folder, clusters_to_ignore, params):
    pass
