# -*- coding: utf-8 -*-
# Author: YeXiaona
# Date  : 2020-12-09

import pandas as pd
import numpy as np


def read_data(in_dir, version = "V1", feature_types=['exp','ppi','seq','go','pathway']):

    exp_file = "%s/SL/%s/Feature/genefea1_exp.csv" % (in_dir, version)
    ppi_file = "%s/SL/%s/Feature/genefea2_ppi.csv" % (in_dir, version)
    seq_file = "%s/SL/%s/Feature/genefea3_seq.csv" % (in_dir, version)
    go_file = "%s/SL/%s/Feature/genefea4_go.csv" % (in_dir, version)
    pathway_file = "%s/SL/%s/Feature/genefea5_pathway.csv" % (in_dir, version)
    extract_file = "%s/SL/%s/Label/genegene_extract.csv" % (in_dir, version)

    exp_data = pd.read_csv(exp_file)
    ppi_data = pd.read_csv(ppi_file)
    seq_data = pd.read_csv(seq_file)
    go_data = pd.read_csv(go_file)
    pathway_data = pd.read_csv(pathway_file)
    extract_data = pd.read_csv(extract_file)

    # print("==========ALL Data Loading==========")
    data = []
    for i, row in extract_data.iterrows():
        if row['gene1'] != row['gene2']:

            gene1_exp, gene2_exp = None, None
            if 'exp' in feature_types:
                gene1_exp = exp_data[exp_data['gene'] == row['gene1']].iloc[:, 1:].values[0]
                gene2_exp = exp_data[exp_data['gene'] == row['gene2']].iloc[:, 1:].values[0]

            gene1_ppi, gene2_ppi = None, None
            if 'ppi' in feature_types:
                gene1_ppi = ppi_data[ppi_data['gene'] == row['gene1']].iloc[:, 1:].values[0]
                gene2_ppi = ppi_data[ppi_data['gene'] == row['gene2']].iloc[:, 1:].values[0]

            gene1_seq, gene2_seq = None, None
            if 'seq' in feature_types:
                gene1_seq = seq_data[seq_data['gene'] == row['gene1']].iloc[:, 1:].values[0]
                gene2_seq = seq_data[seq_data['gene'] == row['gene2']].iloc[:, 1:].values[0]

            gene1_go, gene2_go = None, None
            if 'go' in feature_types:
                gene1_go = go_data[go_data['gene'] == row['gene1']].iloc[:, 1:].values[0]
                gene2_go = go_data[go_data['gene'] == row['gene2']].iloc[:, 1:].values[0]

            gene1_pathway, gene2_pathway = None, None
            if 'pathway' in feature_types:
                gene1_pathway = pathway_data[pathway_data['gene'] == row['gene1']].iloc[:, 1:].values[0]
                gene2_pathway = pathway_data[pathway_data['gene'] == row['gene2']].iloc[:, 1:].values[0]

            label = np.array([row['label']])

            items = [gene1_exp, gene1_ppi, gene1_seq, gene1_go, gene1_pathway,
                     gene2_exp, gene2_ppi, gene2_seq, gene2_go, gene2_pathway, label]
            selected_items = [item for item in items if item is not None]
            sample = np.concatenate(selected_items)

            data.append(sample)
    data = np.array(data)

    # print(data.shape)
    # print("==========Finish Load==========")
    return data

def read_exp_data(in_dir, version = "V1"):
    exp_file = "%s/SL/%s/Feature/genefea1_exp.csv" % (in_dir, version)
    extract_file = "%s/SL/%s/Label/genegene_extract.csv" % (in_dir, version)

    exp_data = pd.read_csv(exp_file)
    extract_data = pd.read_csv(extract_file)

    # print("==========EXP Data Loading==========")
    data = []
    for i, row in extract_data.iterrows():
        if row['gene1'] != row['gene2']:
            data.append(np.hstack((
                exp_data[exp_data['gene'] == row['gene1']].iloc[:, 1:].values[0],
                exp_data[exp_data['gene'] == row['gene2']].iloc[:, 1:].values[0],
                row['label']
            )))
    data = np.array(data)

    # print(data.shape)
    # print("==========Finish Load==========")
    return data

def read_ppi_data(in_dir, version = "V1"):
    ppi_file = "%s/SL/%s/Feature/genefea2_ppi.csv" % (in_dir, version)
    extract_file = "%s/SL/%s/Label/genegene_extract.csv" % (in_dir, version)

    ppi_data = pd.read_csv(ppi_file)
    extract_data = pd.read_csv(extract_file)

    # print("==========PPI Data Loading==========")
    data = []
    for i, row in extract_data.iterrows():
        if row['gene1'] != row['gene2']:
            data.append(np.hstack((
                ppi_data[ppi_data['gene'] == row['gene1']].iloc[:, 1:].values[0],
                ppi_data[ppi_data['gene'] == row['gene2']].iloc[:, 1:].values[0],
                row['label']
            )))
    data = np.array(data)

    # print(data.shape)
    # print("==========Finish Load==========")
    return data

def read_seq_data(version="V1"):
    seq_file = "Datasets/SL/" + version + "/Feature/genefea3_seq.csv"
    extract_file = "Datasets/SL/" + version + "/Label/genegene_extract.csv"

    seq_data = pd.read_csv(seq_file)
    extract_data = pd.read_csv(extract_file)

    print("==========SEQ Data Loading==========")
    data = []
    for i, row in extract_data.iterrows():
        if row['gene1'] != row['gene2']:
            data.append(np.hstack((
                seq_data[seq_data['gene'] == row['gene1']].iloc[:, 1:].values[0],
                seq_data[seq_data['gene'] == row['gene2']].iloc[:, 1:].values[0],
                row['label']
            )))
    data = np.array(data)

    print(data.shape)
    print("==========Finish Load==========")
    return data[:,0:-1],data[:,-1]

def read_go_data(version="V1"):
    go_file = "Datasets/SL/" + version + "/Feature/genefea4_go.csv"
    extract_file = "Datasets/SL/" + version + "/Label/genegene_extract.csv"

    go_data = pd.read_csv(go_file)
    extract_data = pd.read_csv(extract_file)

    print("==========GO Data Loading==========")
    data = []
    for i, row in extract_data.iterrows():
        if row['gene1'] != row['gene2']:
            data.append(np.hstack((
                go_data[go_data['gene'] == row['gene1']].iloc[:, 1:].values[0],
                go_data[go_data['gene'] == row['gene2']].iloc[:, 1:].values[0],
                row['label']
            )))
    data = np.array(data)

    print(data.shape)
    print("==========Finish Load==========")
    return data[:,0:-1],data[:,-1]

def read_pathway_data(version="V1"):
    pathway_file = "./SL_data/" + version + "/Feature/genefea5_pathway.csv"
    extract_file = "./SL_data/" + version + "/Label/genegene_extract.csv"

    pathway_data = pd.read_csv(pathway_file)
    extract_data = pd.read_csv(extract_file)

    print("==========PATHWAY Data Loading==========")
    data = []
    for i, row in extract_data.iterrows():
        if row['gene1'] != row['gene2']:
            data.append(np.hstack((
                pathway_data[pathway_data['gene'] == row['gene1']].iloc[:, 1:].values[0],
                pathway_data[pathway_data['gene'] == row['gene2']].iloc[:, 1:].values[0],
                row['label']
            )))
    data = np.array(data)

    print(data.shape)
    print("==========Finish Load==========")
    return data

def read_feature_selected_data(version="V1"):
    print("==========Load Feature Selected Data==========")
    if version == "V1":
        path = "./SL_data/V1/Feature-Selected-500-Data.csv"
    else:
        path = "./SL_data/V2/Feature-Selected-500-Data.csv"

    data = pd.read_csv(path, header=None).values
    print(data.shape)
    print("==========Finish Load==========")
    return data

def read_pair_feature_selected_data(version="V1"):
    print("==========Load Feature Selected Data==========")
    if version == "V1":
        path = "./SL_data/V1/Pair-Feature-Selected-500-Data.csv"
    else:
        path = "./SL_data/V2/Pair-Feature-Selected-500-Data.csv"

    data = pd.read_csv(path, header=None).values
    print(data.shape)
    print("==========Finish Load==========")
    return data

def read_pair_minus_data(version="V1"):
    print("==========Load Feature Selected Data==========")
    if version == "V1":
        path = "./SL_data/V1/PairMinus-Data.csv"
    else:
        path = "./SL_data/V2/PairMinus-500-Data.csv"

    data = pd.read_csv(path, header=None).values
    print(data.shape)
    print("==========Finish Load==========")
    return data

def read_merge_by_pca_data(version="V1"):
    print("==========Load Merged by PCA Data==========")
    if version == "V1":
        path = "./SL_data/V1/MergeByPCA.csv"
    else:
        path = "./SL_data/V2/MergeByPCA.csv"

    data = pd.read_csv(path, header=None).values
    print(data.shape)
    print("==========Finish Load==========")
    return data