from GetResponse import get_response
import pandas as pd
import Executor


prompt = 'Devise 100 finance sentences with sentiment labels indicating positive, negative, and neutral tones. Use a mixture of financial jargon and plain language to construct the sentences, and consider incorporating timely examples from specific subfields or current events within finance for added diversity and relevance. [sentence] | [label]'

for i in range(0, 100):
    print("=============Now generating " + str(i+1) + " dataset=============")
    generated_data_file = 'best_gendata/best100-' + str(i+1) + '.csv'
    generated_data = Executor.execute(prompt)
    generated_data_df = Executor.saveData(generated_data_file, generated_data)
    print(generated_data)