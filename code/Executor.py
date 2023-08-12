from GetResponse import get_response
import sys
import json
import random
import pandas as pd
import re
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AutoModelForSequenceClassification, pipeline
import numpy as np
import scipy.stats

def execute(prompt):
    user_message = [{'content': prompt, 'role': 'user'}]
    response = get_response(user_message)
    return response

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def calStlye(synData_df, realData_df):
    # count the number of sentences in each data
    synData_num = len(synData_df)
    realData_num = len(realData_df)
    # Define the order of the labels
    labels_order = ['positive', 'negative', 'neutral']

    # Get the sentiment probability distributions in the specified order
    try:
        synData_sentiment_distribution = synData_df['label'].value_counts(normalize=True).loc[labels_order].tolist()
        realData_sentiment_distribution = realData_df['label'].value_counts(normalize=True).loc[labels_order].tolist()
        # synData_sentiment_distribution = (synData_df['label'].value_counts()/synData_num).tolist()
        # realData_sentiment_distribution = (realData_df['label'].value_counts()/realData_num).tolist()
        # Calculate the Jensen-Shannon distance for sentiment probability distributions
        JSD = jensen_shannon_distance(synData_sentiment_distribution, realData_sentiment_distribution)
        sentiment_similarity = 1 - JSD

        # Get the unique words in synData
        readlData_words = realData_df['sentence'].str.lower().str.split().explode()
        realData_words_count = len(readlData_words)
        synData_unique_words = set(synData_df['sentence'].str.lower().str.split().sum())
        # Count the frequency of synData_unique_words in realData
        synData_words_appearance = realData_df['sentence'].str.lower().str.split().explode().isin(synData_unique_words).sum()
        term_frequency_in_sentences = synData_words_appearance / realData_words_count
    except KeyError as e:
        print("KeyError: ", e)
        missing_labels = set(synData_df['label'].unique()) - set(labels_order)
        print("The following labels are in 'synData_df' but not in 'labels_order': ", missing_labels)

        sentiment_similarity = 0
        term_frequency_in_sentences = 0
    
    # Calculate the style reward
    weight = 0.5  # adjust this value based on your preference
    style_reward = weight * sentiment_similarity + (1 - weight) * term_frequency_in_sentences
    print("style_reward: ", style_reward)

    return style_reward


def calAccuracy(synData_df, bert_model):
    try:
        # use pretrained bert model to calculate the accuracy
        finbert = BertForSequenceClassification.from_pretrained(bert_model,num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        task = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
        # Process the text data
        sentences = synData_df['sentence'].tolist()
        pred_results = task(sentences)
        generated_labels = synData_df['label'].tolist()

        # Define the mapping from predicted result to integer label
        label_mapping = {"positive": 0, "negative": 1, "neutral": 2}

        # Convert predicted results to integer labels
        predicted_labels = [label_mapping[pred_result['label']] for pred_result in pred_results]
        # Convert llm generated labels to integer labels
        generated_labels = synData_df['label'].map(label_mapping).tolist()
        # generated_labels = synData_df['label'].str.lower().map(label_mapping).tolist()

        # Calculate accuracy_reward
        accuracy_reward = sum(predicted_label == generated_label for predicted_label, generated_label in zip(predicted_labels, generated_labels)) / len(generated_labels)
        print("accuracy_reward: ", accuracy_reward)
    except:
        print("error encountered in calAccuracy")
        accuracy_reward = 0

    return accuracy_reward


def calLexcial(synData_df, fine_stock_terms_df):
    try:
        # Convert all sentences to lower case and split into words
        synData_words = synData_df['sentence'].str.lower().str.split().explode()

        # count the number of words in synData
        synData_words_num = len(synData_words)
        # Count the number of unique words in synData
        synData_unique_words_num = len(set(synData_df['sentence'].str.lower().str.split().sum()))
        # Calculate the lexical diversity
        synData_lexical_diversity = synData_unique_words_num / synData_words_num

        # Count the total number of words in all sentences
        total_word_count = len(synData_words)
        # Count the total number of words in all sentences that are also in fine_stock_terms_df['term']
        term_count = synData_words.isin(fine_stock_terms_df['term'].str.lower()).sum()
        # Calculate the financial terms frequency in sentences
        term_frequency_in_sentences = term_count / total_word_count

        # Calculate the lexical reward
        weight = 0.5  # adjust this value based on your preference
        lexcical_reward = weight * synData_lexical_diversity + (1 - weight) * term_frequency_in_sentences
        print("lexcial_reward: ", lexcical_reward)
    except:
        print("error encountered in calLexcial")
        lexcical_reward = 0

    return lexcical_reward


def calReward(synData_df, realData_df, bert_model, fine_stock_terms_df, w1, w2, w3):
    reward = w1 * calStlye(synData_df, realData_df) + w2 * calAccuracy(synData_df, bert_model) + w3 * calLexcial(synData_df, fine_stock_terms_df)
    return reward

def saveData(synData_file, synData):
    # Process the text data
    # Split the text data into lines
    lines = synData.strip().split('\n')
    data = []
    for line in lines:
        split_line = line.split('|')
        if len(split_line) != 2:
            print(f"Skipping line '{line}' because it does not contain exactly one '|'")
            continue
        num_sentence, sentiment = map(str.strip, line.split('|'))
        sentence = re.sub(r"^\d+[.)]\s*", "", num_sentence)
        sentiment = sentiment.lower()
        data.append([sentence, sentiment])

    # Convert to DataFrame
    synData_df = pd.DataFrame(data, columns=['sentence', 'label'])
    # Save to CSV
    synData_df.to_csv(synData_file, index=False)
    
    return synData_df

def test_Executor():
    prompt_file = 'generated_data/generated_prompt.csv'
    synData_file = 'generated_data/generated_synData.csv'
    realData_file = '../dataset/FinancialPhraseBank.csv'
    bert_model = 'bert-models/finbert-phrasebank'
    finterms_file = '../dataset/finterms.csv'
    stockterms_file = '../dataset/stockterms.csv'
    
    prompt_df = pd.read_csv(prompt_file)
    prompt = prompt_df['prompt'].iloc[-1]
    synData = execute(prompt)
    print(synData)
    synData_df = saveData(synData_file, synData)

    realData_df = pd.read_csv(realData_file, encoding='latin1')
    fineterms_df = pd.read_csv(finterms_file)
    stockterms_df = pd.read_csv(stockterms_file)
    fineterms_df = fineterms_df.rename(columns={'finterm': 'term'})
    stockterms_df = stockterms_df.rename(columns={'stockterm': 'term'})
    fine_stock_terms_df = pd.concat([fineterms_df, stockterms_df], ignore_index=True)

    w1 = 0.2
    w2 = 0.5
    w3 = 0.3

    reward = calReward(synData_df, realData_df, bert_model, fine_stock_terms_df, w1, w2, w3)
    print("reward: ", reward)

if __name__ == "__main__":
    test_Executor()
    