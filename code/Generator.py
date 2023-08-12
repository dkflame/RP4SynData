from GetResponse import get_response
import sys
import json
import random
import pandas as pd
import Executor

def generate(prompt):
    user_message = [{'content': prompt, 'role': 'user'}]
    generated_prompt = get_response(user_message)
    return generated_prompt

# def randomReward(generated_prompt):
#     # randomly generate a reward
#     reward = random.uniform(-1, 1)
#     return reward

def getRewardfromExecutor(prompt, synData_file, realData_df, bert_model, fine_stock_terms_df, w1, w2, w3):
    synData = Executor.execute(prompt)
    synData_df = Executor.saveData(synData_file, synData)
    reward = Executor.calReward(synData_df, realData_df, bert_model, fine_stock_terms_df, w1, w2, w3)
    return reward

if __name__ == "__main__":
    new_start = True
    start_num = 0
    epochs = 100
    prompt_file = 'generated_data/generated_prompt.csv'
    synData_file = 'generated_data/generated_synData_init.csv'
    bert_model = 'bert-models/finbert-phrasebank'
    realData_file = '../dataset/FinancialPhraseBank.csv'
    finterms_file = '../dataset/finterms.csv'
    stockterms_file = '../dataset/stockterms.csv'

    prompt_df = pd.DataFrame(columns=['prompt', 'reward'])
    realData_df = pd.read_csv(realData_file, encoding='latin1')
    fineterms_df = pd.read_csv(finterms_file)
    stockterms_df = pd.read_csv(stockterms_file)
    fineterms_df = fineterms_df.rename(columns={'finterm': 'term'})
    stockterms_df = stockterms_df.rename(columns={'stockterm': 'term'})
    fine_stock_terms_df = pd.concat([fineterms_df, stockterms_df], ignore_index=True)

    w1 = 0.2
    w2 = 0.5
    w3 = 0.3

    print("====================Initialized====================")
    init_prompt = "Your task is to generate a prompt for an LLM to let it generate 100 Finance text sentences with labels, where the labels are positive, negative, and neutral. Let the LLM generate the sentences and labels with the format [sentence] | [label]. Let the LLM only outputs the sentences and labels only. For your response text, only output the generated prompt. Don't output any other words."

    if new_start:
        generated_prompt = generate(init_prompt)
        reward = getRewardfromExecutor(generated_prompt, synData_file, realData_df, bert_model, fine_stock_terms_df, w1, w2, w3)
        # save generated_prompt and reward to a csv file
        new_row = pd.DataFrame({'prompt': [generated_prompt], 'reward': [reward]})
        print(new_row.to_string(index=False))
        prompt_df = pd.concat([prompt_df, new_row], ignore_index=True)
        prompt_df.to_csv(prompt_file, index=False)
    
    prompt_df = pd.read_csv(prompt_file)

    for i in range(start_num, epochs):
        synData_file = 'generated_data/generated_synData_' + str(i+1) + '.csv'
        print("====================Current Epoch: ", i+1, "====================")
        prompt = "Assume you are a Reinforce Agent. Your current state is this whole prompt you are receiving, and your action is the prompt you are generating this time. You receive a reward for each action, which is ranging from 0 to 1. Take an action to generate a better prompt with different words to recent generated prompts. Below are the recently prompts and rewards: \n" + prompt_df.to_string(index=False)
        step_prompt =  init_prompt + prompt
        print(step_prompt)
        generated_prompt = generate(step_prompt)
        reward = getRewardfromExecutor(generated_prompt, synData_file, realData_df, bert_model, fine_stock_terms_df, w1, w2, w3)
        # save generated_prompt and reward to a csv file
        new_row = pd.DataFrame({'prompt': [generated_prompt], 'reward': [reward]})
        print(new_row.to_string(index=False))
        prompt_df = pd.concat([prompt_df, new_row], ignore_index=True)
        prompt_df.to_csv(prompt_file, index=False)
    
    print("====================Best Result====================")
    best_prompt_reward = prompt_df.loc[prompt_df['reward'].idxmax()]
    print(best_prompt_reward.to_string(index=False))