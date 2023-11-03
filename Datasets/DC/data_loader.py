import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

def load_data(in_dir, cell_line_name = "all", score = "S", type = 'clf', n_class = 2, feature_types=['mf','pp','exp','cl']):

    ### 读取文件
    feature_info = {}

    ## 分子指纹
    finger_file = "%s/DC/drugfeature1_finger_extract.csv" % in_dir
    MF = pd.read_csv(finger_file)
    # 处理分子指纹的表头，便于操作
    column_name = list(MF.columns)
    column_name[0] = "drug_id"
    MF.columns = column_name
    feature_info['mf'] = (MF.shape[1] - 1) * 2


    ## 物化性质
    # 未标准化的物化性质
    phychem_file = "%s/DC/drugfeature2_phychem_extract/drugfeature2_phychem_extract.csv" % in_dir
    # 已标准化的物化性质
    # phychem_file = "%s/DC/drugfeature2_phychem_extract/drugfeature2_phychem_normalize.csv" % in_dir
    PP = pd.read_csv(phychem_file)
    feature_info['pp'] = (PP.shape[1] - 1) * 2

    ## 基因表达谱
    cell_line_path = "%s/DC/drugfeature3_express_extract/" % in_dir
    cell_line_files = ["A-673","A375","A549","HCT116","HS 578T","HT29","LNCAP","LOVO","MCF7","PC-3","RKO","SK-MEL-28","SW-620","VCAP"]
    cell_line_dict = {"A-673":'A673',"A375":'A375',"A549":'A549',"HCT116":'HCT116',"HS 578T":'HS578T',"HT29":"HT29",
                    "LNCAP":"LNCAP","LOVO":'LOVO',"MCF7":'MCF7',"PC-3":'PC3',"RKO":'RKO',"SK-MEL-28":'SKMEL28',"SW-620":'SW620',"VCAP":'VCAP'}
    # 根据参数读取相应细胞系的基因表达
    GP_dict = None
    if cell_line_name == "all":
        GP_dict = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line)) for cell_line in cell_line_files}
    elif type(cell_line_name) is list:
        GP_dict = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line)) for cell_line in cell_line_name}
    elif type(cell_line_name) is str:
        GP_dict = {cell_line_name: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))

    ## 细胞系基因表达谱
    cell_line_feature = "Datasets/DC/cell-line-feature_express_extract.csv"
    CL = pd.read_csv(cell_line_feature)
    feature_info['exp'] = CL.shape[0] * 2
    feature_info['cl'] = CL.shape[0]

    ## 样本数据
    extract_file = "Datasets/DC/drugdrug_extract.csv"
    extract = pd.read_csv(extract_file, usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11])
    # 字符类标号数值化
    label = pd.Categorical(extract["label"])
    extract["label"] = label.codes + 1
    if n_class == 2:
        extract.loc[extract["label"] == 1, "label"] = 0
        extract.loc[extract["label"] == 2, "label"] = 1


    ### 拼接特征
    ## 根据参数读取相应细胞系的药物组合
    # drug_comb是从drugdrug_extract.csv中抽取出来的所需细胞系的药物配对信息，是extract的子集或者全部，[cell_line_name,drugA_id,drugB_id,label]
    drug_comb = None
    if cell_line_name == "all":
        drug_comb = extract
    else:
        if type(cell_line_name) is list:
            drug_comb = extract.loc[extract["cell_line_name"].isin(cell_line_name)]
        else:
            drug_comb = extract.loc[extract["cell_line_name"] == cell_line_name]

    ## 获取新数据的行数和列数
    n_sample = drug_comb.shape[0]
    ### 这里要人工设置特征总长度！！！
    # n_feature = ((MF.shape[1]-1) + (PP.shape[1]-1) + 978) * 2 + 978 + 2
    n_feature = 0
    for feature in feature_types:
        n_feature += feature_info[feature]
    if type == 'all':
        n_feature += 2
    else:
        n_feature += 1

    drug_comb.index = range(n_sample)
    data = np.zeros((n_sample, n_feature))
    for i in range(n_sample):
        drugA_id = drug_comb.at[i,"drug_row_cid"]
        drugB_id = drug_comb.at[i,"drug_col_cid"]

        drugA_MF, drugB_MF = None, None
        if 'mf' in feature_types:
            drugA_MF = get_finger(MF, drugA_id)
            drugB_MF = get_finger(MF, drugB_id)

        drugA_PP, drugB_PP = None, None
        if 'pp' in feature_types:
            drugA_PP = get_phychem(PP, drugA_id)
            drugB_PP = get_phychem(PP, drugB_id)

        drugA_GP, drugB_GP = None, None
        if 'exp' in feature_types:
            cell_line_name = drug_comb.at[i, "cell_line_name"]
            drugA_GP = get_express(GP_dict[cell_line_name], drugA_id)
            drugB_GP = get_express(GP_dict[cell_line_name], drugB_id)

        drug_CL = None
        if 'cl' in feature_types:
            drug_CL = get_cell_feature(CL, cell_line_dict[cell_line_name])

        soft_label = np.array([drug_comb.at[i, score]]) if type in ['reg', 'all'] else None
        label = np.array([drug_comb.at[i, 'label']]) if type in ['clf', 'all'] else None

        items = [drugA_MF, drugA_PP, drugA_GP, drugB_MF, drugB_PP, drugB_GP, drug_CL, soft_label, label]
        selected_items = [item for item in items if item is not None]
        sample = np.concatenate(selected_items)

        data[i] = sample

    return data



def data_loader_balance(in_dir, cell_line_name = "all", type = 'major'):

    ### 读取文件

    ## 基因表达谱
    cell_line_path = "%s/DC/drugfeature3_express_extract/" % in_dir
    cell_line_files = ["A-673","A375","A549","HCT116","HS 578T","HT29","LNCAP","LOVO","MCF7","PC-3","RKO","SK-MEL-28","SW-620","VCAP"]

    # 根据参数读取相应细胞系的基因表达
    GP_dict = None
    if cell_line_name == "all":
        GP_dict = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line)) for cell_line in cell_line_files}
    elif type(cell_line_name) is list:
        GP_dict = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line)) for cell_line in cell_line_name}
    elif type(cell_line_name) is str:
        GP_dict = {cell_line_name: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))

    ## 样本数据
    extract_file = "Datasets/DC/drugdrug_extract.csv"
    extract = pd.read_csv(extract_file, usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11])

    ## 新标签标注
    score_index = [i for i in range(3, 8)]
    scores = extract.iloc[:, score_index]

    score_label = scores.apply(lambda x: (x >= 0).astype(int))
    score_label['major_label'] = score_label.apply(lambda x: x.mode()[0], axis=1)

    mean_label = scores.mean(axis = 1)
    score_label['mean_score'] = mean_label
    score_label['mean_label'] = mean_label.apply(lambda x: np.where(x >= 0, 1, 0))

    new_extract = pd.concat([extract, score_label], axis=1)
    new_extract_file = "Datasets/DC/drugdrug_extract_new.csv"
    new_extract.to_csv(new_extract_file)

    ### 拼接特征
    ## 根据参数读取相应细胞系的药物组合
    # drug_comb是从drugdrug_extract.csv中抽取出来的所需细胞系的药物配对信息，是extract的子集或者全部，[cell_line_name,drugA_id,drugB_id,label]
    drug_comb = None
    if cell_line_name == "all":
        drug_comb = new_extract
    else:
        if type(cell_line_name) is list:
            drug_comb = new_extract.loc[new_extract["cell_line_name"].isin(cell_line_name)]
        else:
            drug_comb = new_extract.loc[new_extract["cell_line_name"] == cell_line_name]

    ## 获取新数据的行数和列数
    n_sample = drug_comb.shape[0]
    n_feature = 978 * 2 + 1
    drug_comb.index = range(n_sample)
    data = np.zeros((n_sample, n_feature))
    for i in range(n_sample):
        drugA_id = drug_comb.at[i,"drug_row_cid"]
        drugB_id = drug_comb.at[i,"drug_col_cid"]

        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_GP = get_express(GP_dict[cell_line_name],drugA_id)
        drugB_GP = get_express(GP_dict[cell_line_name],drugB_id)

        label = None
        if type == 'major':
            label = drug_comb.at[i, 'major_label']
        elif type == 'mean':
            label = drug_comb.at[i, 'mean_label']
        else:
            print("Unknown label type!!!")

        sample = np.hstack((drugA_GP, drugB_GP, label))
        data[i] = sample
    return data




def data_loader_new(in_dir, cell_line_name = "all", type = 'major'):

    ### 读取文件

    ## 基因表达谱
    cell_line_path = "%s/DC/drugfeature3_express_extract/" % in_dir
    cell_line_files = ["A-673","A375","A549","HCT116","HS 578T","HT29","LNCAP","LOVO","MCF7","PC-3","RKO","SK-MEL-28","SW-620","VCAP"]

    # 根据参数读取相应细胞系的基因表达
    GP_dict = None
    if cell_line_name == "all":
        GP_dict = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line)) for cell_line in cell_line_files}
    elif type(cell_line_name) is list:
        GP_dict = {cell_line: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line)) for cell_line in cell_line_name}
    elif type(cell_line_name) is str:
        GP_dict = {cell_line_name: pd.read_csv("{}{}.csv".format(cell_line_path, cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))

    ## 样本数据
    extract_file = "Datasets/DC/drugdrug_extract.csv"
    extract = pd.read_csv(extract_file, usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11])

    ## 新标签标注
    score_index = [i for i in range(3, 8)]
    scores = extract.iloc[:, score_index]

    score_label = scores.apply(lambda x: (x >= 0).astype(int))
    score_label['major_label'] = score_label.apply(lambda x: x.mode()[0], axis=1)

    mean_label = scores.mean(axis=1)
    score_label['mean_score'] = mean_label
    score_label['mean_label'] = mean_label.apply(lambda x: np.where(x >= 0, 1, 0))

    new_extract = pd.concat([extract, score_label], axis=1)
    new_extract_file = "Datasets/DC/drugdrug_extract_new.csv"
    new_extract.to_csv(new_extract_file)

    ### 拼接特征
    ## 根据参数读取相应细胞系的药物组合
    # drug_comb是从drugdrug_extract.csv中抽取出来的所需细胞系的药物配对信息，是extract的子集或者全部，[cell_line_name,drugA_id,drugB_id,label]
    drug_comb = None
    if cell_line_name == "all":
        drug_comb = new_extract
    else:
        if type(cell_line_name) is list:
            drug_comb = new_extract.loc[new_extract["cell_line_name"].isin(cell_line_name)]
        else:
            drug_comb = new_extract.loc[new_extract["cell_line_name"] == cell_line_name]

    ## 获取新数据的行数和列数
    n_sample = drug_comb.shape[0]
    n_feature = 978 * 2 + 1
    drug_comb.index = range(n_sample)
    data = np.zeros((n_sample, n_feature))
    for i in range(n_sample):
        drugA_id = drug_comb.at[i,"drug_row_cid"]
        drugB_id = drug_comb.at[i,"drug_col_cid"]

        cell_line_name = drug_comb.at[i, "cell_line_name"]
        drugA_GP = get_express(GP_dict[cell_line_name],drugA_id)
        drugB_GP = get_express(GP_dict[cell_line_name],drugB_id)

        label = None
        if type == 'major':
            label = drug_comb.at[i, 'major_label']
        elif type == 'mean':
            label = drug_comb.at[i, 'mean_label']
        else:
            print("Unknown label type!!!")

        GP_1 = drugA_GP + drugB_GP
        GP_2 = abs(drugA_GP - drugB_GP)

        sample = np.hstack((GP_1, GP_2, label))
        data[i] = sample
    return data



def get_finger(finger,drug_id):
    drug_finger=finger.loc[finger['drug_id']==drug_id]
    drug_finger=np.array(drug_finger)
    drug_finger=drug_finger.reshape(drug_finger.shape[1])[1:]
    # print(drug_finger.shape)
    return drug_finger

def get_phychem(phychem,drug_id):
    drug_phychem=phychem.loc[phychem["cid"]==drug_id]
    # print(drug_phychem)
    drug_phychem=np.array(drug_phychem)
    drug_phychem=drug_phychem.reshape(drug_phychem.shape[1])[1:]
    # print(drug_phychem.shape)
    return drug_phychem

def get_express(express, drug_id):
    drug_express = express[str(drug_id)]
    drug_express = np.array(drug_express)
    return drug_express

def get_cell_feature(feature, cell_line_name):
    cell_feature = feature[str(cell_line_name)]
    cell_feature = np.array(cell_feature)
    return cell_feature

# data=load_data(cell_line_name="all")
# print(data.shape)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# x=data[:,0:-1]
# y=data[:,-1]
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
# # rf=RandomForestClassifier(n_estimators=10)
# dt=DecisionTreeClassifier()
# dt.fit(x_train,y_train)
# y_pred=dt.predict(x_test)
# print(classification_report(y_test,y_pred))