import pandas as pd
import numpy as np
import math

CONFIG = {
    'NUM_TURNS_TO_CLASSIFY': 1137,
    'NUM_CONTEXT_SAMPLES': 1,
    'MODEL_NAME': 'gpt-3.5-turbo',
    'PRINT_PROMPT_BEFORE_SENDING': False,
    'log_empty_context': False,
}


def get_turns_and_labels(n_context_samples):
    x_data = []
    y_data = []
    zero_context = 0
    non_zero_context = 0
    data = pd.read_csv('incoming_base_spect.csv', encoding='latin1')
    dialogue_nums = np.unique(data['dialogue_num'].to_numpy())
    for dialogue_num in dialogue_nums[:CONFIG['NUM_TURNS_TO_CLASSIFY']]:
        sub_df = data.loc[data['dialogue_num'] == dialogue_num]
        relevant_turn_ids = np.unique(
            sub_df.loc[(sub_df['dialogue_act'] != 'Other') & (sub_df['dialogue_act'] != "0")]['turn_id'].to_numpy())
        if relevant_turn_ids.shape[0] == 0:
            continue
        for turn_id in relevant_turn_ids:
            relevant_rows = sub_df.loc[sub_df['turn_id'] == turn_id]
            dialogue_act = np.delete(np.unique(relevant_rows['dialogue_act'].to_numpy()),
                                     np.where(np.unique(relevant_rows['dialogue_act'].to_numpy()) == 'Other'))[0]
            speaker = np.unique(relevant_rows['emitter'].to_numpy())[0]
            text = ' '.join(str(text) for text in relevant_rows['text'].tolist())
            y_data.append((speaker, text, dialogue_act))
            idx_min = relevant_rows.index.min() - n_context_samples
            if idx_min not in list(sub_df.index):
                idx_min = sub_df.index.min()
            idc = [i for i in range(idx_min, relevant_rows.index.min())]
            context_rows = data.iloc[idc]
            context_speakers = context_rows['emitter'].tolist()
            context_text = context_rows['text'].tolist()
            context_dialogue_act = context_rows['dialogue_act'].tolist()

            context_dialogue_act.reverse()

            if len(context_dialogue_act) != 0:
                if context_dialogue_act[0] == 'Other':
                    i = 0
                    while (i < len(context_dialogue_act)) and (context_dialogue_act[i] == 'Other'):
                        i += 1
                    if i < len(context_dialogue_act) and context_dialogue_act[i] != 'Other':
                        j = 0
                        while context_dialogue_act[j] == 'Other':
                            context_dialogue_act[j] = context_dialogue_act[i]
                            j += 1

                context_dialogue_act.reverse()
                non_zero_context += 1
                x_data.append([(s, context_text[i], context_dialogue_act[i]) for i, s in enumerate(context_speakers)])
            else:
                zero_context += 1
                if CONFIG['log_empty_context']:
                    print("wow")
                    print(context_speakers)
                    print(context_text)
                    print('ydata', y_data[-1])
                    print('dialogue num',dialogue_num)
                    print('turn id',turn_id)
                x_data.append([("None", "", "Start")])



    print("zero_context", zero_context)
    print("non_zero_context", non_zero_context)
    return x_data, y_data


if __name__ == '__main__':
    x_d, y_d = get_turns_and_labels(CONFIG['NUM_CONTEXT_SAMPLES'])
    count_cond = {}
    count_y = {}
    count_x = {}
    for x in x_d:
        dialogue_act_curr=x[-1][2]
        count_x[dialogue_act_curr] = count_x.get(dialogue_act_curr, 0)+1
        count_cond[dialogue_act_curr] = count_cond.get(dialogue_act_curr, {})
    for y in y_d:
        count_y[y[2]] = count_y.get(y[2], 0)+1



    for idx in range(len(x_d)):
        dialogue_act_curr = x_d[idx][-1][2]
        dialogue_act_next = y_d[idx][2]
        count_cond[dialogue_act_curr][dialogue_act_next] = count_cond[dialogue_act_curr].get(dialogue_act_next, 0) + 1


    print("count_y",count_y)
    import collections
    count_cond_sorted={key:collections.OrderedDict(sorted(value.items())) for key,value in count_cond.items()}

    print("count_cond",count_cond_sorted)
    print("count_x",count_x)
    correct_predictions_using_x=sum(max(val.values()) for val in count_cond.values())
    print("correct_predictions_using_x",correct_predictions_using_x)
    print("accuracy", correct_predictions_using_x/sum(sum(val.values()) for val in count_cond.values()))
    # # Introduce 'Other' dialogue act
    # count_cond['Other'] = {}
    #
    #
    # # Update count_y for 'Other' dialogue act
    # count_y['Other'] = len(y_data) - sum(count_y.values())
    #
    # for dialogue_act_next in count_y:
    #     tot_to_dialogue_act_next=0
    #     for dicts in  count_cond.values():
    #         print(dicts)
    #         tot_to_dialogue_act_next +=dicts.get('dialogue_act_next',0)
    #     count_cond['Other'][dialogue_act_next] = count_y[dialogue_act_next]-tot_to_dialogue_act_next

    # Calculating entropy for count_y
    total_samples = sum(count_y.values())
    entropy_ind = 0
    for count_value in count_y.values():
        probability = count_value / total_samples
        if probability > 0:  # Check for zero probability
            entropy_ind -= probability * math.log(probability, 2)

    # Calculating conditioned entropy for count_cond
    conditioned_entropy = 0
    for inner_dict in count_cond.values():
        total_inner_samples = sum(inner_dict.values())
        inner_entropy = 0
        if total_inner_samples != 0:  # Check for zero total_inner_samples
            for count_value in inner_dict.values():
                probability = count_value / total_inner_samples
                if probability > 0:  # Check for zero probability
                    inner_entropy -= probability * math.log(probability, 2)
        conditioned_entropy += inner_entropy * (total_inner_samples / total_samples)

    print("Entropy on count_y:", entropy_ind)
    print("Conditioned entropy on count_cond:", conditioned_entropy)

