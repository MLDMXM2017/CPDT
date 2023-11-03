import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

# 未标准化的物化性质
phychem_file="Datasets/DC/drugfeature2_phychem_extract/drugfeature2_phychem_extract.csv"
# 已标准化的物化性质
# phychem_file="drugfeature2_phychem_extract\drugfeature2_phychem_normalize.csv"
finger_file="Datasets/DC/drugfeature1_finger_extract.csv"
cell_line_path="Datasets/DC/drugfeature3_express_extract/"
cell_line_files=["A-673","A375","A549","HCT116","HS 578T","HT29","LNCAP","LOVO","MCF7","PC-3","RKO","SK-MEL-28","SW-620","VCAP"]
cell_line_dict={"A-673":'A673',"A375":'A375',"A549":'A549',"HCT116":'HCT116',"HS 578T":'HS578T',"HT29":"HT29",
                "LNCAP":"LNCAP","LOVO":'LOVO',"MCF7":'MCF7',"PC-3":'PC3',"RKO":'RKO',"SK-MEL-28":'SKMEL28',"SW-620":'SW620',"VCAP":'VCAP'}
cell_line_feature_express_extract_file = './data/cell-line-feature_express_extract.csv'
extract_file="Datasets/DC/drugdrug_extract.csv"
cell_line_feature="Datasets/DC/cell-line-feature_express_extract.csv"

finger = pd.read_csv(finger_file)
# print(finger.shape[1])

def load_data2(cell_line_name="all",score="S"):
    '''
    cell_line_name参数用来控制所选择的细胞系
    可以给入的参数类型为list和str
        若"all"，选取全部细胞系（默认）
        若细胞系名，选取单个细胞系，如"HT29"
        若list,选取部分细胞系，如 ["HT29","A375"]
    '''
    # 读取药物组合信息，药物物化性质，药物分子指纹
    extract=pd.read_csv(extract_file,usecols=[3,4,5,6,7,8,9,10,11])
    phychem=pd.read_csv(phychem_file)
    finger=pd.read_csv(finger_file)
    cell_feature=pd.read_csv(cell_line_feature)
    # 处理分子指纹的表头，便于操作
    column_name=list(finger.columns)
    column_name[0]="drug_id"
    finger.columns=column_name
    # 字符类标号数值化
    label=pd.Categorical(extract["label"])
    extract["label"]=label.codes+1
    # 根据参数读取相应细胞系的基因表达
    if cell_line_name=="all":
        all_express={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_files}
    elif type(cell_line_name) is list:
        all_express={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_name}
    elif type(cell_line_name) is str:
        all_express={cell_line_name:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))
    # 根据参数读取相应细胞系的药物组合
    # drug_comb是从drugdrug_extract.csv中抽取出来的所需细胞系的药物配对信息，是extract的子集或者全部，[cell_line_name,drugA_id,drugB_id,label]
    drug_comb=None
    if cell_line_name=="all":
        drug_comb=extract
    else:
        if type(cell_line_name) is list:
            drug_comb=extract.loc[extract["cell_line_name"].isin(cell_line_name)]
        else:
            drug_comb=extract.loc[extract["cell_line_name"]==cell_line_name]
    # 获取新数据的行数和列数
    n_sample=drug_comb.shape[0]
    n_feature=((phychem.shape[1]-1)+(finger.shape[1]-1)+978)*2+978+2
    drug_comb.index=range(n_sample)
    data=np.zeros((n_sample,n_feature))
    for i in range(n_sample):
        drugA_id=drug_comb.at[i,"drug_row_cid"]
        drugB_id=drug_comb.at[i,"drug_col_cid"]
        drugA_finger=get_finger(finger,drugA_id)
        drugB_finger=get_finger(finger,drugB_id)
        drugA_phychem=get_phychem(phychem,drugA_id)
        drugB_phychem=get_phychem(phychem,drugB_id)
        cell_line_name=drug_comb.at[i,"cell_line_name"]
        drugA_express=get_express(all_express[cell_line_name],drugA_id)
        drugB_express=get_express(all_express[cell_line_name],drugB_id)
        # drugA=np.hstack((drugA_finger,drugA_phychem,drugA_express))
        # drugB=np.hstack((drugB_finger,drugB_phychem,drugB_express))
        feature=get_cell_feature(cell_feature,cell_line_dict[cell_line_name])
        soft_label=drug_comb.at[i,score]
        label=drug_comb.at[i,'label']
        # sample=np.hstack((drugA,drugB,label))
        # sample=np.hstack((drugA_finger,drugA_phychem,drugA_express,drugB_finger,drugB_phychem,drugB_express,feature,label))
        # sample=np.hstack((drugA_finger,drugA_phychem,drugB_finger,drugB_phychem,feature,label))
        sample=np.hstack((drugA_finger,drugA_phychem,drugA_express,drugB_finger,drugB_phychem,drugB_express,feature,soft_label,label))
        data[i]=sample
        # break
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

def get_express(express,drug_id):
    drug_express=express[str(drug_id)]
    drug_express=np.array(drug_express)
    # print(drug_express.shape)
    return drug_express

def get_cell_feature(feature,cell_line_name):
    # print(feature.head())
    # print(cell_line_name)
    cell_feature=feature[str(cell_line_name)]
    cell_feature=np.array(cell_feature)
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