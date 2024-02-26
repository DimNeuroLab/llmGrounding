import pandas as pd
import numpy as np
import math

CONFIG = {
    'NUM_TURNS_TO_CLASSIFY': 1137,
    'NUM_CONTEXT_SAMPLES': 1,
    'MODEL_NAME': 'gpt-3.5-turbo',
    'PRINT_PROMPT_BEFORE_SENDING': False,
}

def get_turns_and_labels(n_context_samples):
    x_data = []
    y_data = []

    data = pd.read_csv('incoming_base_spect.csv', encoding='latin1')
    dialogue_nums = np.unique(data['dialogue_num'].to_numpy())
    for dialogue_num in dialogue_nums[:CONFIG['NUM_TURNS_TO_CLASSIFY']]:
        sub_df = data.loc[data['dialogue_num'] == dialogue_num]
        relevant_turn_ids = np.unique(sub_df.loc[(sub_df['dialogue_act'] != 'Other') & (sub_df['dialogue_act'] != "0")]['turn_id'].to_numpy())
        if relevant_turn_ids.shape[0] == 0:
            continue
        for turn_id in relevant_turn_ids:
            relevant_rows = sub_df.loc[sub_df['turn_id'] == turn_id]
            dialogue_act = np.delete(np.unique(relevant_rows['dialogue_act'].to_numpy()),
                                     np.where(np.unique(relevant_rows['dialogue_act'].to_numpy()) == 'Other'))[0]
            speaker = np.unique(relevant_rows['emitter'].to_numpy())[0]
            text = ' '.join(str(text) for text in relevant_rows['text'].tolist())
            y_data.append((speaker, text, dialogue_act))
            idx_min = relevant_rows.index.min()-n_context_samples
            if idx_min not in list(sub_df.index):
                idx_min = sub_df.index.min()
            idc = [i for i in range(idx_min, relevant_rows.index.min())]
            context_rows = data.iloc[idc]
            context_speakers = context_rows['emitter'].tolist()
            context_text = context_rows['text'].tolist()
            context_dialogue_act = context_rows['dialogue_act'].tolist()
            print(context_dialogue_act)
            print("context_rows['dialogue_act']")
            print(context_rows['dialogue_act'])
            print("context_text")
            print(context_text)
            context_dialogue_act.reverse()

            if len(context_dialogue_act)!=0:
                if context_dialogue_act[0] == 'Other':
                    i=0
                    while (i <len(context_dialogue_act))  and (context_dialogue_act[i]=='Other'):
                        i+=1
                    if i <len(context_dialogue_act) and context_dialogue_act[i]!='Other':
                       j=0
                       while context_dialogue_act[j] == 'Other':
                        context_dialogue_act[j]=context_dialogue_act[i]
                        j+=1

                context_dialogue_act.reverse()


            x_data.append([(s, context_text[i],context_dialogue_act[i]) for i, s in enumerate(context_speakers)])


    return x_data, y_data

if __name__ == '__main__':
    x_data, y_data = get_turns_and_labels(CONFIG['NUM_CONTEXT_SAMPLES'])
    count = {}
    count_ind = {}
    
    for y in y_data:
        count[y[2]] = count.get(y[2], {})
        count_ind[y[2]] = count_ind.get(y[2], 0)

    for idx in range(len(y_data)-1):
        dialogue_act_curr = y_data[idx][2]
        dialogue_act_next = y_data[idx+1][2]
        count[dialogue_act_curr][dialogue_act_next] = count[dialogue_act_curr].get(dialogue_act_next, 0) + 1
        count_ind[dialogue_act_curr] += 1

    print(count_ind)

    print(count)
    # # Introduce 'Other' dialogue act
    # count['Other'] = {}
    #
    #
    # # Update count_ind for 'Other' dialogue act
    # count_ind['Other'] = len(y_data) - sum(count_ind.values())
    #
    # for dialogue_act_next in count_ind:
    #     tot_to_dialogue_act_next=0
    #     for dicts in  count.values():
    #         print(dicts)
    #         tot_to_dialogue_act_next +=dicts.get('dialogue_act_next',0)
    #     count['Other'][dialogue_act_next] = count_ind[dialogue_act_next]-tot_to_dialogue_act_next



    # Calculating entropy for count_ind
    total_samples = sum(count_ind.values())
    entropy_ind = 0
    for count_value in count_ind.values():
        probability = count_value / total_samples
        if probability > 0:  # Check for zero probability
            entropy_ind -= probability * math.log(probability, 2)

    # Calculating conditioned entropy for count
    conditioned_entropy = 0
    for inner_dict in count.values():
        total_inner_samples = sum(inner_dict.values())
        inner_entropy = 0
        if total_inner_samples != 0:  # Check for zero total_inner_samples
            for count_value in inner_dict.values():
                probability = count_value / total_inner_samples
                if probability > 0:  # Check for zero probability
                    inner_entropy -= probability * math.log(probability, 2)
        conditioned_entropy += inner_entropy * (total_inner_samples / total_samples)

    print("Entropy on count_ind:", entropy_ind)
    print("Conditioned entropy on count:", conditioned_entropy)
    print("Count:", count)
    print("Count_ind:", count_ind)
