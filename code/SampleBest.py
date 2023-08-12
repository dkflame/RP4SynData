import pandas as pd
import argparse

def sample(output_csv_path, positive_num, negative_num, neutral_num):
    # 类别和数量的映射
    label_counts = {"positive": positive_num, "negative": negative_num, "neutral": neutral_num}
    # 文件名的模板
    file_template = 'best_gendata/best100-{}.csv'
    # 文件的数量
    file_count = 100
    # 用于存储所有数据的 DataFrame
    df_list = []

    # 读入所有的数据
    for i in range(1, file_count + 1):
        file_name = file_template.format(i)
        df = pd.read_csv(file_name)
        df_list.append(df)

    # 将所有的 DataFrame 合并为一个
    df_all = pd.concat(df_list, ignore_index=True)

    # 去重
    df_all.drop_duplicates(subset=['sentence'], keep='first', inplace=True)

    # 用于存储最终结果的 DataFrame
    df_final = pd.DataFrame()

    # 对每一类别进行处理
    for label, count in label_counts.items():
        # 获取该类别的数据
        df_label = df_all[df_all['label'] == label]
        # 如果数量不足，则取出所有该类别的数据
        if df_label.shape[0] < count:
            df_final = pd.concat([df_final, df_label], ignore_index=True)
        else:
            # 否则，随机抽取指定数量的数据
            df_sample = df_label.sample(n=count, random_state=730)
            df_final = pd.concat([df_final, df_sample], ignore_index=True)

    # 打乱顺序
    df_final = df_final.sample(frac=1).reset_index(drop=True)

    # 保存到文件
    df_final.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    # output_csv_file， positive_num, negative_num, neutral_num 由终端输入
    parser = argparse.ArgumentParser(description="Sample from CSV files")
    parser.add_argument('--outfilename', type=str, required=True, help="The output CSV file")
    parser.add_argument('--positive', type=int, required=True, help="The number of positive samples")
    parser.add_argument('--negative', type=int, required=True, help="The number of negative samples")
    parser.add_argument('--neutral', type=int, required=True, help="The number of neutral samples")
    args = parser.parse_args()

    output_csv_path = 'best_gendata/' + args.outfilename
    sample(output_csv_path, args.positive, args.negative, args.neutral)