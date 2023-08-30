import csv
import glob
import os
import random
import shutil

import numpy as np
import pandas as pd
# -*- coding: utf-8 -*-

def rename_file():
    # root = '/home/postgraduate/sunxinhuan/PycharmProject/data/Kidney_lc-1_TCGA_RESULTS_DIRECTORY/KIRC'
    # for file in os.listdir(root):
    #     if file != 'Step_2.csv' and file != 'process_list_autogen.csv':
    #         file_path = os.path.join(root,file)
    #         for tcag in os.listdir(file_path):
    #             if 'KIRC' in tcag:
    #                 old_path = os.path.join(file_path,tcag)
    #                 new_path = os.path.join(file_path , tcag[4:])
    #                 print(new_path)
    #                 os.rename(old_path,new_path)
    csv_path = '/home/postgraduate/sunxinhuan/PycharmProject/data/Kidney_lc-1_TCGA_RESULTS_DIRECTORY/KIRC/Step_2.csv'
    data = pd.read_csv(csv_path)
    # print(data['slide_id'])
    for i in range(len(data['slide_id'])):
        data['slide_id'][i] = data['slide_id'][i][4:]
    data.to_csv(csv_path,index=False, encoding='utf-8')


def to_csv(args):
    f = open(args.step_3_path, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['case_id', 'slide_id', 'label'])
    f.close()

def to_heatmap_csv(args):
    f = open(args.heatmap_demo_dataset_path, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['slide_id', 'label'])
    f.close()
def write_heatmap_csv(heatmap_data,args):
    with open(args.heatmap_demo_dataset_path, mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        for i in heatmap_data:
            wf.writerow(i)
def write_csv(step_3_data,args):
    with open(args.step_3_path, mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        for i in step_3_data:
            wf.writerow(i)

def read_csv(args):
    label = args.label
    clinical_data = pd.read_table(args.clinical_path)
    step_2_data = pd.read_csv(args.step_2_path)
    case_id = clinical_data['CDE_ID:2003301']
    slide_id = step_2_data['slide_id']
    step_3_data = []
    for i in range(len(case_id)):
        for j in range(len(slide_id)):
            if case_id[i] in slide_id[j]:
                step_3_data.append([case_id[i],slide_id[j],label])
    return step_3_data


def write_heatmap_demo_dataset():
    root = '/home/postgraduate/sunxinhuan/PycharmProject/data/Kidney_TCGA_RESULTS_DIRECTORY/Step_3.csv'
    data = pd.read_csv(root)
    slide_id = data['slide_id']
    heatmap_data = []
    for i in range(len(slide_id)):
        heatmap_data.append([data['slide_id'][i],data['label'][i]])
    return heatmap_data

def get_heatmap_data():
    root = '/remote-home/sunxinhuan/PycharmProject/data/kidney/Kidney'
    heatmap_path = '/remote-home/sunxinhuan/PycharmProject/CLAM-master/heatmaps/demo/slides/'
    heatmap_data = []
    for file in os.listdir(root):
        label = file
        file_path = os.path.join(root,file)
        for svs in os.listdir(file_path):
            rand = random.randint(0,15)
            if rand == 0:
                svs_name = svs.split('.')[0] + '.' + svs.split('.')[1]
                svs_path = os.path.join(file_path, svs)
                shutil.copy(svs_path,heatmap_path)
                heatmap_data.append([svs_name,label])
    return heatmap_data

def ccrcc_label(args):
    clinical_data = pd.read_table(args.clinical_path)
    return clinical_data
    # case_id = clinical_data['case_submitter_id']
    # label = clinical_data['tumor_grade']

def CAMELYON16_step_3(args):
    root = '/remote-home/sunxinhuan/PycharmProject/data/CAMELYON16_RESULTS_1023/CAMELYON16_512_1_RESULTS_DIRECTORY'
    step_3_data = []
    to_csv(args)
    for label in os.listdir(root):
        if label != 'testing':
            tif_path = os.path.join(root,label,'masks')
            for tif in os.listdir(tif_path):
                tif_name = tif.split('.')[0]
                step_3_data.append([tif_name,tif_name,label])
        else:
            test_patient = pd.read_csv(os.path.join(root,label,'reference.csv'))
            test_patient = np.array(test_patient)
            print(test_patient[0,:])
            for i in range(0,test_patient.shape[0]):
                step_3_data.append([test_patient[i,0],test_patient[i,0],test_patient[i,1].lower()])

    print(len(step_3_data))
    write_csv(step_3_data,args)

def CAMELYON16_heatmap(args):
    to_heatmap_csv(args)
    root = '/remote-home/sunxinhuan/PycharmProject/data/test/CAMELYON16'
    heatmap_data = []
    for patient in os.listdir(root):
        patient_name = patient.split('.')[0]
        heatmap_data.append([patient_name,'tumor'])
    write_heatmap_csv(heatmap_data,args)

def sgz_step_3(args):
    step_sgz_exlx_path = '/remote-home/sunxinhuan/PycharmProject/data/Lung/cohort.xlsx'
    step_sgz_exlx = pd.read_excel(step_sgz_exlx_path)
    step_sgz_exlx = np.array(step_sgz_exlx)
    print(step_sgz_exlx.shape)
    step_3 = []
    # step_3_path = '/remote-home/sunxinhuan/PycharmProject/data/step_3_sgz.csv'
    to_csv(args)
    root = '/remote-home/sunxinhuan/PycharmProject/data/Lung/Lung_512_1_RESULTS_DIRECTORY'
    for group in os.listdir(root):
        group_path = os.path.join(root,group)
        for file in os.listdir(group_path):
            if file == 'stitches':
                file_path = os.path.join(group_path,file)
                for patient in os.listdir(file_path):
                    patient_name = patient.split('.')[0]
                    # if patient_name not in step_sgz_exlx:
                    #     step_3.append(patient_name)
                    for i in range(step_sgz_exlx.shape[0]):
                        if patient_name in step_sgz_exlx[i,3] or patient_name in step_sgz_exlx[i,2]:
                            step_3.append([step_sgz_exlx[i,1],patient_name,group])
    print(step_3)
    print(len(step_3))
    write_csv(step_3,args)

def SGZ_Step_3(args):
    SGZ_Step_3_path = '/remote-home/sunxinhuan/PycharmProject/data/SGZ_0129/SGZ_Step_3.xlsx'
    SGZ_Step_3_data = pd.read_excel(SGZ_Step_3_path)
    SGZ_Step_3_data = np.array(SGZ_Step_3_data)
    root = '/remote-home/sunxinhuan/PycharmProject/data/SGZ_0129/SZG_512_123_FEATURES_DIRECTORY/Three_subtypes_featurs/h5_files'
    Step_3 = []
    to_csv(args)
    for patient in os.listdir(root):
        patient_name = patient.split('.')[0]
        for i in range(SGZ_Step_3_data.shape[0]):
            if patient_name == str(SGZ_Step_3_data[i,1]):
                Step_3.append([SGZ_Step_3_data[i,0],SGZ_Step_3_data[i,1],SGZ_Step_3_data[i,2]])
    Step_3 = np.array(Step_3)
    write_csv(Step_3,args)


def KIRC_Step_3(args):
    clinical_path = '/remote-home/sunxinhuan/PycharmProject/data/kidney/C_MAE/KIRC/clinical.xlsx'
    clinical_data = pd.read_excel(clinical_path)
    clinical_data = np.array(clinical_data)

    svs_path = '/remote-home/sunxinhuan/PycharmProject/data/kidney/C_MAE/KIRC/Stage'
    step_3= []
    to_csv(args)

    for i in range(clinical_data.shape[0]):
        for grade in os.listdir(svs_path):
            grade_path = os.path.join(svs_path,grade)
            for svs in os.listdir(grade_path):
                svs_name = svs.split('.')[0] + '.' +svs.split('.')[1]
                if clinical_data[i,1] in svs_name:
                    step_3.append([clinical_data[i,1],svs_name,grade])
    step_3 = np.array(step_3)
    write_csv(step_3,args)






import argparse
parser = argparse.ArgumentParser(description="make mask")
parser.add_argument("--label", default=None,type=str)
parser.add_argument("--step_2_path", default=None,type=str)
parser.add_argument("--clinical_path", default=None,type=str)
parser.add_argument("--step_3_path", default='/remote-home/sunxinhuan/PycharmProject/data/kidney/C_MAE/KIRC/Step_3.csv',type=str)
parser.add_argument("--heatmap_demo_dataset_path", default='/remote-home/sunxinhuan/PycharmProject/CLAM-master/heatmaps/demo/CAMELYON16_heatmap_demo_dataset.csv',type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    # CAMELYON16_heatmap(args)
    # CAMELYON16_step_3(args)
    # SGZ_Step_3(args)
    KIRC_Step_3(args)
    # SGZ_Step_3(args)

    # rename_file()
    # to_heatmap_csv(args)
    # root = '/remote-home/sunxinhuan/PycharmProject/data/PyCLAM/Kidney_512_2_TCGA_RESULTS_DIRECTORY'
    # to_csv(args)
    # for file in os.listdir(root):
    #     file_path = os.path.join(root,file)
    #     step_2_path = os.path.join(file_path,'Step_2.csv')
    #     # clinical_path = glob.glob( os.path.join(file_path,'*.txt'))
    #     # print(clinical_path[0])
    #     args.label = file
    #     args.step_2_path = step_2_path
    #     # args.clinical_path = clinical_path[0]
    #     # clinical_data = ccrcc_label(args)
    #     # print(clinical_data)
    #     # step_3_data = []
    #     # for patient in os.listdir(os.path.join(file_path,'patches')):
    #     #     patient_name = patient.split('-')[0] + '-' + patient.split('-')[1]
    #     #     for i in range(len(clinical_data['case_submitter_id'])):
    #     #         if patient_name == clinical_data['case_submitter_id'][i]:
    #     #             step_3_data.append([clinical_data['case_submitter_id'][i],patient.split('.')[0],clinical_data['tumor_grade'][i]])
    #     # print(len(step_3_data))
    #     # write_csv(step_3_data, args)
    #     step_3_data = read_csv(args)
    #     print(len(step_3_data))
    #     write_csv(step_3_data,args)
    # heatmap_data = get_heatmap_data()
    # write_heatmap_csv(heatmap_data,args)




