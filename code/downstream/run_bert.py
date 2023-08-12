
import numpy as np
import pandas as pd 
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments, AutoModel, AutoTokenizer
from transformers import pipeline
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import transformers
import argparse
from sklearn.metrics import classification_report



torch.__version__, transformers.__version__

if torch.cuda.is_available():
    # GPU可用
    device = torch.device("cuda:0")  # 将索引号为0的GPU设置为当前设备
    print("Available GPU numbers: ", torch.cuda.device_count())
else:
    # 没有可用的GPU
    device = torch.device("cpu")
    print("No available GPU")

# 示例代码中使用的设备
print("Current Device: ", device)

label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
label_mapping_inverse = {v: k for k, v in label_mapping.items()}


def load_data(synData_file, realData_file, sample_num, random_state=805):
    synData_df = pd.read_csv(synData_file)
    realData_df = pd.read_csv(realData_file, encoding='latin1')

    synData_df = synData_df.dropna(subset=['sentence', 'label'])
    realData_df = realData_df.dropna(subset=['sentence', 'label'])

    synData_df_train, synData_df_val = train_test_split(synData_df, test_size=0.1, random_state=random_state)
    # synData_df_train = synData_df.sample(n=90, random_state=random_state)
    # synData_df_val = synData_df.drop(synData_df_train.index).reset_index(drop=True)

    realData_df_train_val = realData_df.sample(n=sample_num, random_state=random_state)
    realData_df_train, realData_df_val = train_test_split(realData_df_train_val, test_size=0.1, random_state=random_state)
    realData_df_test = realData_df.drop(realData_df_train_val.index).reset_index(drop=True)
    return synData_df_train, synData_df_val, realData_df_train, realData_df_val, realData_df_test

def prepare_data(data_df, tokenizer):
    data_df['label'] = data_df['label'].map(label_mapping)
    dataset = Dataset.from_pandas(data_df)
    dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    return dataset

def load_model(model_file):
    model = BertForSequenceClassification.from_pretrained(model_file,num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_file)
    model.config.id2label = label_mapping_inverse
    model.config.label2id = label_mapping
    return model, tokenizer

def getTrainer(model, Dataset_train, Dataset_val, num_train_epochs):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy' : accuracy_score(predictions, labels)}

    args = TrainingArguments(
            output_dir = 'temp/',
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            # save_strategy='steps',
            # save_steps=150,
            learning_rate=2e-5,
            per_device_train_batch_size=90,
            per_device_eval_batch_size=40,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
    )
    trainer = Trainer(
            model=model,
            args=args,
            train_dataset=Dataset_train,
            eval_dataset=Dataset_val,
            compute_metrics=compute_metrics
    )
    return trainer

def train(model, tokenizer, output_model_path, dataset_train, dataset_val, dataset_test, num_train_epochs):
    trainer = getTrainer(model, dataset_train, dataset_val, num_train_epochs)
    trainer.train()
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    model.eval()
    model_pred = trainer.predict(dataset_val).metrics
    print(model_pred)
    return model

def predict(model, tokenizer, input_data_df, other_label_mapping=None):
    # task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True, padding=True, max_length=128)

    # Process the text data
    sentences = input_data_df['sentence'].tolist()
    pred_results = task(sentences)
    predicted_labels = [pred_result['label'] for pred_result in pred_results]
    gold_labels = input_data_df['label'].tolist()

    # print("Predicted labels: ", predicted_labels)
    # print("Gold labels: ", gold_labels)

    # # Convert labels to integer labels
    # predicted_labels = [label_mapping[pred_result['label']] for pred_result in pred_results]
    # gold_labels = input_data_df['label'].map(label_mapping).tolist()

    # Convert bert label if a mapping is provided
    if other_label_mapping is not None:
        predicted_labels = [other_label_mapping[label] for label in predicted_labels]

    # Calculate accuracy
    accuracy = sum(predicted_label == generated_label for predicted_label, generated_label in zip(predicted_labels, gold_labels)) / len(gold_labels)

    # Calculate precision, recall for each class
    classReport = classification_report(gold_labels, predicted_labels, target_names=['positive', 'negative', 'neutral'])

    return accuracy, classReport


def predict_batch(model, tokenizer, input_data_df, other_label_mapping=None, batch_size=90):
    # 对每一批进行预测
    predictions = []
    for i in range(0, len(input_data_df), batch_size):
        batch = input_data_df.iloc[i:i+batch_size]
        # 对每个句子进行编码，并转换为PyTorch张量
        encoding = tokenizer(batch['sentence'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoding['input_ids'].to(device)

        # 进行预
        with torch.no_grad():
            logits = model(input_ids)[0]
        
        # 使用argmax得到预测的类别
        predicted_classes = torch.argmax(logits, dim=1).tolist()
        predictions.extend(predicted_classes)
    
    # 将字符串标签映射为整数
    gold_labels = input_data_df['label'].map(label_mapping).values

    # 计算精度
    accuracy = accuracy_score(gold_labels, predictions)
    # 计算分类报告
    classReport = classification_report(gold_labels, predictions)
    
    return accuracy, classReport



def predict_onlineBert(model_path, input_data_df):
    task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
    # Process the text data
    sentences = input_data_df['sentence'].tolist()
    pred_results = task(sentences)
    predicted_labels = [pred_result['label'] for pred_result in pred_results]
    gold_labels = input_data_df['label'].tolist()

    # print("Predicted labels: ", predicted_labels)
    # print("Gold labels: ", gold_labels)

    # Calculate accuracy
    accuracy = sum(predicted_label == generated_label for predicted_label, generated_label in zip(predicted_labels, gold_labels)) / len(gold_labels)

    # Calculate precision, recall for each class
    classRreport = classification_report(gold_labels, predicted_labels, target_names=['positive', 'negative', 'neutral'])

    return accuracy, classRreport



if __name__ == "__main__":
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description="Train or Test a BERT model")
    # 添加命令行参数
    parser.add_argument('--trainbySyn', action='store_true', help="Train the model for Synthetic Data")
    parser.add_argument('--trainbyReal', action='store_true', help="Train the model for Real Data")
    parser.add_argument('--resume_modelbySyn_file', type=str, help='The path of the model file to be resumed on training')
    parser.add_argument('--resume_modelbyReal_file', type=str, help='The path of the model file to be resumed on training')
    parser.add_argument('--test', type=str, default=None, help='The model to be tested')
    parser.add_argument('--sample_num', type=int, required=True, help="The number of samples")
    parser.add_argument('--num_train_epochs', type=int, default=100, help='The number of training epochs')
    parser.add_argument('--random_state', type=int, default=805, help='The random state for splitting data')

    # 解析命令行参数
    args = parser.parse_args()
    num_train_epochs = args.num_train_epochs
    sample_num = args.sample_num
    random_state = args.random_state

    synData_file = '../best_gendata/synData' +  str(sample_num) +'.csv'
    realData_file = '../../dataset/FinancialPhraseBank.csv'
    bert_base_model_file = '../bert-models/bert-base-uncased'
    output_modelbySyn_path = 'models/modelbySyn' +  str(sample_num)
    output_modelbyReal_path = 'models/modelbyReal' +  str(sample_num)
    bertTweet_model_file = '../bert-models/bert-Tweet'
    bert_prasebank_model_file = '../bert-models/finbert-phrasebank'


    synData_df_train, synData_df_val, realData_df_train, realData_df_val, realData_df_test = load_data(synData_file, realData_file, sample_num, random_state=random_state)

    if args.trainbySyn:
        print("============Training mode for synData selected============")
        modelbySyn, tokenizerbySyn = load_model(bert_base_model_file)

        synDataset_train = prepare_data(synData_df_train, tokenizerbySyn)
        synDataset_val = prepare_data(synData_df_val, tokenizerbySyn)
        realDataset_train = prepare_data(realData_df_train, tokenizerbySyn)
        realDataset_test = prepare_data(realData_df_test, tokenizerbySyn)

        finetuned_modelbySyn = train(modelbySyn, tokenizerbySyn, output_modelbySyn_path, synDataset_train, synDataset_val, realDataset_test, num_train_epochs)
        # finetuned_modelbySyn = train(modelbySyn, tokenizerbySyn, output_modelbySyn_path, synDataset_train, realDataset_train, realDataset_test, num_train_epochs)

    if args.resume_modelbySyn_file:
        print("============Resuming Training mode for synData selected============")
        modelbySyn, tokenizerbySyn = load_model(args.resume_modelbySyn_file)

        synDataset_train = prepare_data(synData_df_train, tokenizerbySyn)
        synDataset_val = prepare_data(synData_df_val, tokenizerbySyn)
        realDataset_train = prepare_data(realData_df_train, tokenizerbySyn)
        realDataset_test = prepare_data(realData_df_test, tokenizerbySyn)

        # finetuned_modelbySyn = train(modelbySyn, tokenizerbySyn, output_modelbySyn_path, synDataset_train, synDataset_val, realDataset_test, num_train_epochs)
        finetuned_modelbySyn = train(modelbySyn, tokenizerbySyn, output_modelbySyn_path, synDataset_train, realDataset_test, realDataset_test, num_train_epochs)

    if args.trainbyReal:
        print("============Training mode for realData selected============")
        modelbyReal, tokenizerbyReal = load_model(bert_base_model_file)

        realDataset_train = prepare_data(realData_df_train, tokenizerbyReal)
        realDataset_val = prepare_data(realData_df_val, tokenizerbyReal)
        realDataset_test = prepare_data(realData_df_test, tokenizerbyReal)

        finetuned_modelbyReal = train(modelbyReal, tokenizerbyReal, output_modelbyReal_path, realDataset_train, realDataset_val, realDataset_test, num_train_epochs)

    if args.resume_modelbyReal_file:
        print("============Resuming Training mode for realData selected============")
        modelbyReal, tokenizerbyReal = load_model(args.resume_modelbyReal_file)

        realDataset_train = prepare_data(realData_df_train, tokenizerbyReal)
        realDataset_val = prepare_data(realData_df_val, tokenizerbyReal)
        realDataset_test = prepare_data(realData_df_test, tokenizerbyReal)

        finetuned_modelbyReal = train(modelbyReal, tokenizerbyReal, output_modelbyReal_path, realDataset_train, realDataset_val, realDataset_test, num_train_epochs)

    if args.test is not None:
        print("============Testing mode selected============")
        finetuned_modelbySyn = BertForSequenceClassification.from_pretrained(output_modelbySyn_path,num_labels=3)
        tokenizerbySyn = BertTokenizer.from_pretrained(output_modelbySyn_path)
        finetuned_modelbyReal = BertForSequenceClassification.from_pretrained(output_modelbyReal_path,num_labels=3)
        tokenizerbyReal = BertTokenizer.from_pretrained(output_modelbyReal_path)
        bertPhrasebank = BertForSequenceClassification.from_pretrained(bert_prasebank_model_file,num_labels=3)
        tokenizerbyBertPhrasebank = BertTokenizer.from_pretrained(bert_prasebank_model_file)
        bertTweet = BertForSequenceClassification.from_pretrained(bertTweet_model_file,num_labels=3)
        tokenizerbyBertTweet = BertTokenizer.from_pretrained(bertTweet_model_file)
        
        # test_df = realData_df_test.sample(n=sample_num, random_state=random_state)
        test_df = realData_df_test

        # test bert-Tweet on Real data
        if args.test == 'bert-Tweet':
            other_label_mapping = {'NEGATIVE': 'negative', 'NEUTRAL': 'neutral', 'POSITIVE': 'positive'}
            if args.test == 'bert-Tweet':
                accuracybyTweet, classReportbyTweet = predict(bertTweet, tokenizerbyBertTweet, test_df, other_label_mapping)
                print("Bert Tweet Predicting Accuracy: ", accuracybyTweet)
                print("Bert Tweet Classification Report: \n", classReportbyTweet)


        # test bert-Phrasebank on Real data
        if args.test == 'bert-Phrasebank':
            accuracybyPhrasebank, classReportbyPhrasebank = predict(bertPhrasebank, tokenizerbyBertPhrasebank, test_df)
            print("Bert Phrasebank Predicting Accuracy: ", accuracybyPhrasebank)
            print("Bert Phrasebank Classification Report: \n", classReportbyPhrasebank)

        if args.test == 'bert-Syn':
            # test bert-Syn on Real data
            accuracybySyn, classReportbySyn = predict(finetuned_modelbySyn, tokenizerbySyn, test_df)
            print("Finetuned Bert on Synthetic Data Predicting Accuracy: ", accuracybySyn)
            print("Finetuned Bert on Synthetic Data Classification Report: \n", classReportbySyn)

        if args.test == 'bert-Real':
            # test bert-Real on Real data
            accuracybyReal, classReportbyReal = predict(finetuned_modelbyReal, tokenizerbyReal, test_df)
            print("Finetuned Bert on Real Data Predicting Accuracy: ", accuracybyReal)
            print("Finetuned Bert on Real Data Classification Report: \n", classReportbyReal)

