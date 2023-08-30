import os
import pandas as pd

import os
import argparse
# 这个是上一步生成的文件



parser = argparse.ArgumentParser()

# 数据集所在根目录

parser.add_argument('--csv_path', type=str,
                    default=None)
parser.add_argument('--patch_path', type=str,
                    default=None)
parser.add_argument('--save_path', type=str,
                    default=None)


def main(args):
    df = pd.read_csv(args.csv_path)
    ids1 = [i[:-4] for i in df.slide_id]
    ids2 = [i[:-3] for i in
            os.listdir(args.patch_path)]
    df['slide_id'] = ids1
    ids = df['slide_id'].isin(ids2)
    sum(ids)
    df.loc[ids].to_csv(args.save_path, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    RESULTS_DIRECTORY_path = '/remote-home/sunxinhuan/PycharmProject/data/CAMELYON16_RESULTS_1023/512_20×_RESULTS_DIRECTORY_test'
    for who_ in os.listdir(RESULTS_DIRECTORY_path):
        who_path = os.path.join(RESULTS_DIRECTORY_path,who_)
        for file in os.listdir(who_path):
            if file == 'process_list_autogen.csv':
                args.csv_path = os.path.join(who_path,file)
            if file == 'patches':
                args.patch_path = os.path.join(who_path,file)
        args.save_path = os.path.join(who_path,'Step_2.csv')
        main(args)




