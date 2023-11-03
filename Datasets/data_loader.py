
from Datasets.DC.data_loader import load_data as load_data_DC
from Datasets.DC.data_loader import data_loader_balance as load_data_DC_b
from Datasets.DC.data_loader import data_loader_new as load_data_DC_n
from Datasets.SL.data_loader_sl import read_data as load_data_SL
from Datasets.SL.data_loader_sl import read_exp_data as load_data_SL_exp
from Datasets.SL.data_loader_sl import read_ppi_data as load_data_SL_ppi


def load_data(in_dir, file_name):
    data_name, feature_comb = file_name.split('_')
    feature_types = feature_comb.split('+')
    if data_name == 'DC':
        return load_data_DC(in_dir, feature_types=feature_types)
    elif data_name == 'SL':
        return load_data_SL(in_dir, feature_types=feature_types)










