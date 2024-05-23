import numpy as np
import os
import pandas as pd
import sys
import argparse

pd.options.mode.copy_on_write = True
sample_summary_df = pd.DataFrame()

locus_parameters = {
    "TP73_1": [8, 9, 10, 11, 12, 13, 14],
    "F5_2": [0, 1, 2],
    "C1orf132": [2, 3, 4, 5, 6, 7, 8, 9],
    "FHL2": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "ZEB2": [0, 1, 2],
    "LHFPL4": [0, 1],
    "GRM2_9": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "cg23500537": [1, 2, 3],
    "ELOVL2_6": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "cg10804656": [12, 13, 14, 15, 16, 17, 18],
    "ABCC4": [0, 1, 2],
    "CCDC102B": [0, 1, 2, 3],
    "MBP": [0, 1, 2, 3],
    "EPHX3": [0, 1, 2, 3, 4],
    "SLC12A5": [2, 3, 4, 5, 6, 7, 8],
    "SAMD10_2": [0, 1, 2, 3, 4, 5]
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


def hist_pre_processing(files, args, sample_name):
    """
    :param files: files should be related to one person only, otherwise the returned data will contain a mix
     of different samples
    :param loci_of_focus:
    :return:
    """
    sample_by_loci = {}
    for hist_file in files:
        hist_file = open(hist_file)
        loci = hist_file.readline().split('\n')[0]
        loci_len = len(locus_parameters[loci])
        if loci not in all_possible_patterns:
            all_possible_patterns[loci] = set()
            calc_all_possible_patterns("T" * min(10, loci_len), 0, loci_len, loci)
        df = pd.read_csv(hist_file, sep='\t')
        all_reads_count = df['Count'].sum()
        if all_reads_count < int(args.reads_threshold):
            print(f'\nThe file {hist_file.name} have less than {args.reads_threshold} reads({all_reads_count}).\n',
                  file=sys.stderr)
            # continue
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
        boot_rows = np.zeros((len(new_df.index) * args.num_sub_samples, len(list(all_possible_patterns[loci]))))
        row_counter = 0
        for index, row in new_df.iterrows():
            mul_prob_values = row.values / row['sum']
            for i in range(args.num_sub_samples):
                out = np.random.multinomial(args.reads_per_sample, mul_prob_values[:-1])
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


def concat(args):
    list_of_markers_to_concat = [
        ('ELOVL2_6', 'C1orf132'),
        ('ELOVL2_6', 'C1orf132', 'FHL2'),
        ('ELOVL2_6', 'C1orf132', 'FHL2', 'CCDC102B'),
        ('ELOVL2_6', 'C1orf132', 'CCDC102B')
    ]
    for markers_to_concat in list_of_markers_to_concat:
        final_df = pd.DataFrame()
        marker_name = '_'.join(markers_to_concat)
        print("Concatinating ", marker_name, file=sys.stderr)
        empty_markers_counter = {}
        for marker in markers_to_concat:
            df = pd.read_csv(args.output_directory + '/' + marker + "_" + str(args.reads_per_sample) + ".txt", sep='\t')
            if len(markers_to_concat) > 3:
                for sample_name in df[df['total_reads_origin'] == 0]['sample_name'].unique():
                    if sample_name in empty_markers_counter:
                        empty_markers_counter[sample_name] += 1
                    else:
                        empty_markers_counter[sample_name] = 1
            if final_df.empty:
                final_df = df
                continue
            else:
                df.drop(columns=['sample_name'], inplace=True)
            final_df = pd.concat([final_df, df], axis=1)
        for sample_name in empty_markers_counter:
            if empty_markers_counter[sample_name] > 0:
                final_df = final_df[final_df['sample_name'] != sample_name]
        if len(empty_markers_counter) > 3:
            final_df = final_df.fillna(final_df.mean(numeric_only=True))
        final_df.to_csv(args.output_directory + '/' + marker_name + "_" + str(args.reads_per_sample) + ".txt", index=False, sep='\t')


def main(args):
    for marker in locus_parameters:
        all_together_df = pd.DataFrame()
        df = pd.read_csv(args.summary_file, sep='\t')
        if df.empty or 'Sample #' not in df.columns:
            df = pd.read_csv(args.summary_file)
            if df.empty or 'Sample #' not in df.columns:
                print("Summary file is empty or in wrong format.\n", file=sys.stderr)
                continue
        df = df[['Sample #', 'Gene', 'Sample']]
        sample_summary_df = df
        sample_summary_df["Gene"] = sample_summary_df["Gene"].str.strip()
        sample_summary_df = sample_summary_df[sample_summary_df['Gene'] == marker]
        sample_summary_df.sort_values(by=['Sample'], inplace=True)

        all_dfs = pd.DataFrame()
        for sample in sample_summary_df.groupby('Sample'):
            sample_files = sample[1]['Sample #'].tolist()
            sample_name = sample[0]
            if sample_name == "ddw":
                continue
            hists = os.listdir(args.hists_path)
            hists = [args.hists_path + '/' + hist for hist in hists]
            hist_files = [file for file in hists if '.hist' in file and int(file.split('/')[-1].split('_')[1]) in sample_files]
            dfs = hist_pre_processing(hist_files, args, sample_name)
            if len(dfs) > 0 and not all_together_df.empty and sample_name in all_together_df[
                "sample_name"].unique():
                read_counts = dfs[marker][1]
                prev_read_count = all_together_df[all_together_df["sample_name"] == sample_name][
                    "total_reads_origin"].unique().item()
                if read_counts > prev_read_count:
                    all_together_df = all_together_df[all_together_df["sample_name"] != sample_name]
                else:
                    print("Sample " + sample_name + " is a duplicate that already exists.", file=sys.stderr)
                    continue
            if len(dfs) == 0:
                print("Sample " + sample_name + " contains no good results.", file=sys.stderr)
                continue
            df = dfs[marker][0]
            df['sample_name'] = sample_name
            df['total_reads_origin'] = dfs[marker][1]
            all_dfs = pd.concat([all_dfs, df], ignore_index=True)
        if all_dfs.empty:
            print("Marker ", marker, " is empty.", file=sys.stderr)
            continue
        all_together_df = pd.concat([all_together_df, all_dfs], ignore_index=True)
        if all_together_df.empty:
            print("Marker ", marker, " is missing or doesn't pass the threshold.", file=sys.stderr)
            continue
        all_together_df.sort_values(by='sample_name', inplace=True)
        all_together_df.reset_index(inplace=True)
        all_together_df.drop(columns=["index"], inplace=True)
        all_together_df.to_csv(
            args.output_directory + '/' + marker + '_' + str(args.reads_per_sample) + ".txt",
            sep='\t',
            index=False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hists_path')
    parser.add_argument('-sum', '--summary_file')
    parser.add_argument('-out', '--output_directory', default='./')
    parser.add_argument('-t', '--reads_threshold', default=1000, type=int)
    parser.add_argument('-rps', '--reads_per_sample', default=8192, type=int)
    parser.add_argument('-nss', '--num_sub_samples', default=128, type=int)
    args = parser.parse_args()
    if not args.hists_path or not args.summary_file:
        print("Please provide --hists_path and --summary_file arguments", file=sys.stderr)
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)
    main(args)
    concat(args)


