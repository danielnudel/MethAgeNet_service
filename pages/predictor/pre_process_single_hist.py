import numpy as np
import pandas as pd
import os
import argparse
import sys
from copy import deepcopy

pd.options.mode.copy_on_write = True
sample_summary_df = pd.DataFrame()

locus_parameters = {
    # "TP73_1": [8, 9, 10, 11, 12, 13, 14],
    # "F5_2": [0, 1, 2],
    "C1orf132": [2, 3, 4, 5, 6, 7, 8, 9],
    "FHL2": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    # "ZEB2": [0, 1, 2],
    # "LHFPL4": [0, 1],
    # "GRM2_9": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # "cg23500537": [1, 2, 3],
    "ELOVL2_6": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    # "cg10804656": [12, 13, 14, 15, 16, 17, 18],
    # "ABCC4": [0, 1, 2],
    "CCDC102B": [0, 1, 2, 3]
    # "MBP": [0, 1, 2, 3],
    # "EPHX3": [0, 1, 2, 3, 4],
    # "SLC12A5": [2, 3, 4, 5, 6, 7, 8],
    # "SAMD10_2": [0, 1, 2, 3, 4, 5]
}

all_possible_patterns = dict()


def calc_all_possible_patterns(label, index, site_len, loci):
    if index > min(10, site_len - 1):
        return
    t_label = label[:index] + "T" + label[index + 1:]
    c_label = label[:index] + "C" + label[index + 1:]
    all_possible_patterns[loci].add(t_label)
    all_possible_patterns[loci].add(c_label)
    calc_all_possible_patterns(t_label, index + 1, site_len, loci)
    calc_all_possible_patterns(c_label, index + 1, site_len, loci)


for loci in locus_parameters:
    all_possible_patterns[loci] = set()
    loci_len = len(locus_parameters[loci])
    calc_all_possible_patterns("T" * min(10, loci_len), 0, loci_len, loci)


READS_THRESHOLD = 1000
NUM_SUB_SAMPLES = 128
READS_PER_SAMPLE = 8192
def hist_pre_processing(hist_file):
    """
    :param files: files should be related to one person only, otherwise the returned data will contain a mix
     of different samples
    :param loci_of_focus:
    :return:
    """
    sample_by_loci = {}
    loci = hist_file.readline().split('\n')[0]
    if loci not in locus_parameters:
        return (pd.DataFrame(), None, loci)
    loci_len = len(locus_parameters[loci])
    if loci not in all_possible_patterns:
        all_possible_patterns[loci] = set()
        calc_all_possible_patterns("T" * min(10, loci_len), 0, loci_len, loci)
    df = pd.read_csv(hist_file, sep='\t')
    all_reads_count = df['Count'].sum()
    if all_reads_count < int(READS_THRESHOLD):
        print(f'\nThe file have less than {READS_THRESHOLD} reads({all_reads_count}).\n', file=sys.stderr)
    columns_to_ignore = ["# of A", "# of C", "# of T", "# of G", "# of sites"]
    for i in range(6, len(df.columns)):
        if i-6 not in locus_parameters[loci]:
            columns_to_ignore.append(df.columns[i])
    df.drop(columns_to_ignore, axis='columns', inplace=True)
    df = df[~(df == '-').any(axis=1)]
    if not df.empty:
        if loci not in sample_by_loci:
            sample_by_loci[loci] = df
        else:
            sample_by_loci[loci] = pd.concat((sample_by_loci[loci], df))

    df = sample_by_loci[loci]
    total_reads_counts = df['Count'].sum()
    df['read'] = df[df.columns[1:]].apply(lambda x: ''.join(x), axis=1)
    df.reset_index(drop=True, inplace=True)
    df = df[['Count', 'read']]
    df['total'] = df.groupby(['read'])['Count'].transform('sum')
    df = df.drop_duplicates(subset=['read'])[['total', 'read']]
    df.sort_values(by='total', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    df['name'] = loci

    new_df = df.pivot_table(index='name', columns='read', values='total')
    current_columns = new_df.columns
    columns_to_add = [col for col in sorted(list(all_possible_patterns[loci])) if col not in current_columns]
    if len(columns_to_add) != 0:
        df_to_add = pd.DataFrame(columns=columns_to_add)
        new_df = pd.concat((new_df, df_to_add))
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    new_df.fillna(0, inplace=True)
    new_df['sum'] = new_df.sum(axis=1)
    boot_rows = np.zeros((len(new_df.index) * NUM_SUB_SAMPLES, len(list(all_possible_patterns[loci]))))
    row_counter = 0
    for index, row in new_df.iterrows():
        mul_prob_values = row.values / row['sum']
        for i in range(NUM_SUB_SAMPLES):
            out = np.random.multinomial(READS_PER_SAMPLE, mul_prob_values[:-1])
            try:
                boot_rows[row_counter, :] = out
            except:
                return (pd.DataFrame(), None, loci)
            row_counter += 1
    boots = pd.DataFrame(boot_rows, columns=new_df.columns[:-1])

    locus_sites = locus_parameters[loci]
    fixed_columns = boots.columns
    for i in range(len(locus_sites) + 1):
        columns_of_interest = [c for c in fixed_columns if c.count('C') == i]
        boots["C_count_" + str(i)] = boots[columns_of_interest].sum(axis=1)
    for site in range(len(locus_sites)):
        columns_of_interest = [c for c in fixed_columns if c[site] == 'C']
        boots["site_" + str(site + 1)] = boots[columns_of_interest].sum(axis=1)
    columns_order = [c for c in boots.columns if "site_" in c] +\
                    [c for c in boots.columns if "C_count_" in c] +\
                    sorted(list(all_possible_patterns[loci]), reverse=True)
    boots = boots[columns_order]
    return (boots, total_reads_counts, loci)


def hist_from_df_pre_processing(df, loci):
    """
    :param files: files should be related to one person only, otherwise the returned data will contain a mix
     of different samples
    :param loci_of_focus:
    :return:
    """
    sample_by_loci = {}
    if loci not in locus_parameters:
        return (pd.DataFrame(), None, loci)
    loci_len = len(locus_parameters[loci])
    if loci not in all_possible_patterns:
        all_possible_patterns[loci] = set()
        calc_all_possible_patterns("T" * min(10, loci_len), 0, loci_len, loci)
    df = df[~(df == '-').any(axis=1)]
    if not df.empty:
        if loci not in sample_by_loci:
            sample_by_loci[loci] = df
        else:
            sample_by_loci[loci] = pd.concat((sample_by_loci[loci], df))

    df = sample_by_loci[loci]
    total_reads_counts = df['Count'].sum()
    df.reset_index(drop=True, inplace=True)
    df = df[['Count', 'read']]
    df['total'] = df.groupby(['read'])['Count'].transform('sum')
    df = df.drop_duplicates(subset=['read'])[['total', 'read']]
    df.sort_values(by='total', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    df['name'] = loci
    print(df)

    new_df = df.pivot_table(index='name', columns='read', values='total')
    current_columns = new_df.columns
    columns_to_add = [col for col in sorted(list(all_possible_patterns[loci])) if col not in current_columns]
    if len(columns_to_add) != 0:
        df_to_add = pd.DataFrame(columns=columns_to_add)
        new_df = pd.concat((new_df, df_to_add))
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    new_df.fillna(0, inplace=True)
    new_df['sum'] = new_df.sum(axis=1)
    boot_rows = np.zeros((len(new_df.index) * NUM_SUB_SAMPLES, len(list(all_possible_patterns[loci]))))
    row_counter = 0
    for index, row in new_df.iterrows():
        mul_prob_values = row.values / row['sum']
        for i in range(NUM_SUB_SAMPLES):
            out = np.random.multinomial(READS_PER_SAMPLE, mul_prob_values[:-1])
            try:
                boot_rows[row_counter, :] = out
            except:
                return (pd.DataFrame(), None, loci)
            row_counter += 1
    boots = pd.DataFrame(boot_rows, columns=new_df.columns[:-1])

    locus_sites = locus_parameters[loci]
    fixed_columns = boots.columns
    for i in range(len(locus_sites) + 1):
        columns_of_interest = [c for c in fixed_columns if c.count('C') == i]
        boots["C_count_" + str(i)] = boots[columns_of_interest].sum(axis=1)
    for site in range(len(locus_sites)):
        columns_of_interest = [c for c in fixed_columns if c[site] == 'C']
        boots["site_" + str(site + 1)] = boots[columns_of_interest].sum(axis=1)
    columns_order = [c for c in boots.columns if "site_" in c] +\
                    [c for c in boots.columns if "C_count_" in c] +\
                    sorted(list(all_possible_patterns[loci]), reverse=True)
    boots = boots[columns_order]
    return (boots, total_reads_counts, loci)

def hist_from_multiple_dfs_pre_processing(files):
    """
    :param files: files should be related to one person only, otherwise the returned data will contain a mix
     of different samples
    :param loci_of_focus:
    :return:
    """
    sample_by_loci = {}
    for hist_file in files:
        loci = hist_file.readline().split('\n')[0]
        if loci not in locus_parameters:
            continue
        loci_len = len(locus_parameters[loci])
        if loci not in all_possible_patterns:
            all_possible_patterns[loci] = set()
            calc_all_possible_patterns("T" * min(10, loci_len), 0, loci_len, loci)
        df = pd.read_csv(hist_file, sep='\t')
        columns_to_ignore = ["# of A", "# of C", "# of T", "# of G", "# of sites"]
        for i in range(6, len(df.columns)):
            if i - 6 not in locus_parameters[loci]:
                columns_to_ignore.append(df.columns[i])
        df.drop(columns_to_ignore, axis='columns', inplace=True)
        df = df[~(df == '-').any(axis=1)]
        if not df.empty:
            if loci not in sample_by_loci:
                sample_by_loci[loci] = df
            else:
                sample_by_loci[loci] = pd.concat((sample_by_loci[loci], df))
    for loci in sample_by_loci:
        df = sample_by_loci[loci]
        total_reads_counts = df['Count'].sum()
        df['read'] = df[df.columns[1:]].apply(lambda x: ''.join(x), axis=1)
        df.reset_index(drop=True, inplace=True)
        df = df[['Count', 'read']]
        df['total'] = df.groupby(['read'])['Count'].transform('sum')
        df = df.drop_duplicates(subset=['read'])[['total', 'read']]
        df.sort_values(by='total', inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)
        df['name'] = loci

        new_df = df.pivot_table(index='name', columns='read', values='total')
        current_columns = new_df.columns
        columns_to_add = [col for col in sorted(list(all_possible_patterns[loci])) if col not in current_columns]
        if len(columns_to_add) != 0:
            df_to_add = pd.DataFrame(columns=columns_to_add)
            new_df = pd.concat((new_df, df_to_add))
        new_df = new_df.reindex(sorted(new_df.columns), axis=1)
        new_df.fillna(0, inplace=True)
        new_df['sum'] = new_df.sum(axis=1)
        boot_rows = np.zeros((len(new_df.index) * NUM_SUB_SAMPLES, len(list(all_possible_patterns[loci]))))
        row_counter = 0
        for index, row in new_df.iterrows():
            mul_prob_values = row.values / row['sum']
            for i in range(NUM_SUB_SAMPLES):
                out = np.random.multinomial(READS_PER_SAMPLE, mul_prob_values[:-1])
                try:
                    boot_rows[row_counter, :] = out
                except:
                    breakpoint()
                row_counter += 1
        boots = pd.DataFrame(boot_rows, columns=new_df.columns[:-1])

        locus_sites = locus_parameters[loci]
        fixed_columns = boots.columns
        for i in range(len(locus_sites) + 1):
            columns_of_interest = [c for c in fixed_columns if c.count('C') == i]
            boots["C_count_" + str(i)] = boots[columns_of_interest].sum(axis=1)
        for site in range(len(locus_sites)):
            columns_of_interest = [c for c in fixed_columns if c[site] == 'C']
            boots["site_" + str(site + 1)] = boots[columns_of_interest].sum(axis=1)
        columns_order = [c for c in boots.columns if "site_" in c] +\
                        [c for c in boots.columns if "C_count_" in c] +\
                        sorted(list(all_possible_patterns[loci]), reverse=True)
        boots = boots[columns_order]
        sample_by_loci[loci] = (boots, total_reads_counts)
    return sample_by_loci


def concat(dict_by_marker):
    final_dfs = {}
    list_of_markers_to_concat = [
        ('ELOVL2_6', 'C1orf132'),
        ('ELOVL2_6', 'C1orf132', 'FHL2'),
        ('ELOVL2_6', 'C1orf132', 'FHL2', 'CCDC102B'),
        ('ELOVL2_6', 'C1orf132', 'CCDC102B')
    ]
    for markers_to_concat in list_of_markers_to_concat:
        final_df = np.array([])
        marker_name = '_'.join(markers_to_concat)
        print("Concatinating ", marker_name, file=sys.stderr)
        for marker in markers_to_concat:
            if marker not in dict_by_marker:
                print(f'Marker {marker} is missing')
                final_df = np.array([])
                break
            if len(final_df) == 0:
                final_df = dict_by_marker[marker][0]
            else:
                final_df = np.concatenate([final_df, dict_by_marker[marker][0]], axis=1)
        if len(final_df) > 0:
            final_dfs[marker_name] = final_df
    return final_dfs

def hist_cohort_pre_processing(files, summary):
    all_together_dict = {}
    summary_copy = deepcopy(summary)
    df = pd.read_csv(summary)
    if df.empty or 'Sample #' not in df.columns:
        df = pd.read_csv(summary_copy, sep='\t')
        if df.empty or 'Sample #' not in df.columns:
            print("Summary file is empty or in wrong format.\n", file=sys.stderr)
            return {}
    df = df[['Sample #', 'Gene', 'Sample']]
    sample_summary_df = df
    sample_summary_df["Gene"] = sample_summary_df["Gene"].str.strip()
    for sample in sample_summary_df.groupby('Sample'):
        sample_files = sample[1]['Sample #'].tolist()
        sample_files = ['Sample_' + str(num) + '_CpG.hist' for num in sample_files]
        sample_name = sample[0]
        files_for_sample = [files[file] for file in files if file in sample_files]
        hists_dict = hist_from_multiple_dfs_pre_processing(files_for_sample)
        hists_dict_all = concat(hists_dict)
        for hist in hists_dict:
            hists_dict_all[hist] = hists_dict[hist][0]
        all_together_dict[sample_name] = hists_dict_all
    return all_together_dict






# if __name__=="__main__":
#     file = open("C:\\Users\\danie\\OneDrive\\Desktop\\MethAgeNet_service\\pages\\predictor\\test\\Sample_106_CpG.hist")
#     a, b, c = hist_pre_processing(file)
#     print(a.empty)
#     print(b)
#     print(c)


