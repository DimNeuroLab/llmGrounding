import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
import time
import openai
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import functools
import threading
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff



CONFIG = {
    'NUM_TURNS_TO_CLASSIFY': 1137, # set to which turn number the script should go
    'NUM_CONTEXT_SAMPLES': 3, # how many of the previous turn should be used as context; if you put -1 all turns will be used
    'MODEL_NAME': 'gpt-3.5-turbo', # type of model to use
    'PRINT_PROMPT_BEFORE_SENDING': False, # only set to true for debug! This will not result in real ChatGPT calls
}


def get_api_key():
    with open('key.txt', 'r') as key_file:
        key = key_file.read()
    return key


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
            x_data.append([(s, context_text[i]) for i, s in enumerate(context_speakers)])
    return x_data, y_data


def evaluate_prompts(y_true, y_pred):
    mapping = {
        'other': 0,
        'offer': 1,
        'counteroffer': 2,
        'accept': 3,
        'refusal': 4
    }
    y_true = [mapping[y] for y in y_true]
    y_pred = [mapping[y] for y in y_pred]
    labels = list(mapping.keys())
    print(classification_report(y_true=y_true, y_pred=y_pred, digits=5, target_names=labels))


def call_chatgpt(prompt):
    conversation = [{'role': 'system', 'content': prompt}]
    print(conversation)
    max_attempts = 5
    for _ in range(max_attempts):
        try:
            time.sleep(1)
            # with timeoutWindows(seconds=30):

            response =timeout(20)(openai.ChatCompletion.create)(
                model=CONFIG['MODEL_NAME'],
                messages=conversation
                )
            print("\n")
            print(response)
            response_parsed = response['choices'][0]['message']['content']

            return response_parsed
        except Exception as ex:
            print('ERROR')
            print(ex)
            response_parsed = -1
            pass
    print('TIME OUT')
    print('TIME OUT')
    print('TIME OUT')
    return response_parsed

def timeout(seconds_before_timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, seconds_before_timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = threading.Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e:
                print('error starting thread')
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco
def parse_response(response):
    if response == 'other':
        return 'other'
    else:
        response = response.split(',')
        response = response[0]
        response = response.replace('[', '')
        response = response.replace(']', '')
        if response.lower() in ['offer', 'counteroffer', 'accept', 'refusal']:
            return response.lower()
        else:
            return 'other'


def build_prompt_to_classify(input_sample, sample_to_classify):
    dialogue = ''
    if len(input_sample)>0:
        for context in input_sample:
                        dialogue += context[0] + ': ' + str(context[1]) + '\n'
    context= sample_to_classify
    dialogue += context[0] + ': ' + context[1] + '\n'
    classdescriptions=["Offer: A proposal to trade resources between players, which isn't related to another offer. Example: Hey anyone have any clay?\n",
    "Counteroffer: A response to another player's offer, proposing a different trade. Example: I can do 1 of each for 2 clay.\n",
    "Accept: Agreeing to an offer or counteroffer made by another player. Example: I can wheat for clay.\n",
    "Refusal: Declining an offer or counteroffer made by another player. Example: No, not interested.\n",
    "Other: Turns or statements that do not involve direct trading, such as discussing game mechanics or making observations about the current state of the game, including questions that aren't offers or counteroffers. Example: What’s up?\n"]
    classdescriptions_text="".join(classdescriptions)
    
    prompt = f"""I will give you a dialogue from a game of Settlers of Catan played by some players, you will need to evaluate the class of the last utterance.


It is very important that you consider what said by each player, which represent their intentions, and the order in which each player spoke. Build (but don't write) the framework of which resources each player wants to trade for giving and which to trade for receiving.
    
The admissible classes of utterances, with definition and examples are:\n
{classdescriptions_text}

Please remember: If an utterance qualifies for the class "Other" but also for one of the other 4 classes, it should then be considered of the other class (not of the class "Other")    

As a suggestiion, please consider these trends:
- After an "Offer" the most likely dialogue act is "Refusal".
Example:
player_blue: "anybody need ore?" (Offer)
player_red: "sry no" (Refusal)

- After a "Counteroffer" the most likely dialogue act is one among "Accept" or "Refusal" or "Counteroffer".
Example:
player_bird: "wheat for clay?" (Offer)
player_whale: "I don't need clay. I can do for wood" (Counteroffer)
player_bird: "I can spare one" (Accept) or player_grey: "Nope" (Refusal) or player_grey: "Anything else than wood?" (Counteroffer)

- After an "Accept" the most likely dialogue act is another "Accept".
Example:
player_tall: "Anyone wants clay?" (Offer)
player_small: "Yes" (Accept)
player_tall: "k" (Accept)

- After a "Refusal" the most kikely dialogue act is another "Refusal".
Example:
player_Mike: "No" (Refusal)
player_Anna: "sry can't" (Refusal)

- After a "Other" the most likely dialogue act is "Offer".
Example:
player_dark: "ahah" (Other)
player_light: "wheat for me?" (Offer)

The dialogue to consider for the prediction of the class of the next utterance:\n
{dialogue}

    
    What is the class of the last utterance of the dialogue?\n
    Very important: please respond with 1 possible continuation in this precise format: [class of utterance]
    """
    return prompt.replace('\t', '')


def write_results_to_file(y_true, y_pred, y_true_parsed, y_pred_parsed):
    ts = str(time.time()).split('.')[0]
    n_turns = str(CONFIG['NUM_TURNS_TO_CLASSIFY'])
    n_context = str(CONFIG['NUM_CONTEXT_SAMPLES'])
    import os
    with open(os.path.basename(__file__)+'_n_turns_' + n_turns + '_n_context_' + n_context +'_' + ts + '.tsv', 'w') as out_file:
        out_file.write('true_act\tpred_act\ttrue_turn\tpred_turn\n')
        for idx, y_t in enumerate(y_true_parsed):
            out_file.write(y_t+'\t')
            out_file.write(y_pred_parsed[idx]+'\t')
            out_file.write(y_true[idx]+'\t')
            out_file.write(y_pred[idx] + '\n')


if __name__ == '__main__':
    openai.api_key = get_api_key()
    x_data, y_data = get_turns_and_labels(n_context_samples=CONFIG['NUM_CONTEXT_SAMPLES'])
    y_pred_all = []
    y_true_all = []
    y_pred_raw = []
    y_true_raw = []
    for idx, sample in enumerate(tqdm(x_data)):
        #if len(sample) > 0:
        prompt = build_prompt_to_classify(sample, y_data[idx])
        if CONFIG['PRINT_PROMPT_BEFORE_SENDING']:
            print(prompt)
            continue
        chat_gpt_output = call_chatgpt(prompt)
        y_pred_raw.append(chat_gpt_output)
        y_pred = parse_response(chat_gpt_output)
        y_pred_all.append(y_pred)
        y_true_all.append(y_data[idx][-1].lower())
        y_true_raw.append('; '.join(y_data[idx]))
        if (idx > 0) & (divmod(idx, 50)[1] == 0):
            print("evaluations at "+str(idx)+" samples")
            evaluate_prompts(y_true_all, y_pred_all)
    if not CONFIG['PRINT_PROMPT_BEFORE_SENDING'] and len(y_pred_raw) > 0:
        write_results_to_file(y_true=y_true_raw,
                              y_pred=y_pred_raw,
                              y_true_parsed=y_true_all,
                              y_pred_parsed=y_pred_all)
        evaluate_prompts(y_true_all, y_pred_all)

