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
    'NUM_CONTEXT_SAMPLES': 1, # how many of the previous turn should be used as context; if you put -1 all turns will be used
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


def build_prompt(input_sample):
    dialogue = ''
    for context in input_sample:
        dialogue += context[1] + '\n' # context[0] + ': ' + context[1] + '\n'
    prompt = f"""I will give you a dialogue from a game of Settlers of Catan, you will need to continue it with 1 possible utterance from 1 player, and 1 class.
     
    The admissible types of utterances, with definition are:\n
    Offer: A proposal to trade resources between players, which isn't related to another offer
    Counteroffer: A response to another player's offer, proposing a different trade.
    Accept: Agreeing to an offer or counteroffer made by another player.
    Refusal: Declining an offer or counteroffer made by another player.
    Other: Turns or statements that do not involve direct trading, such as discussing game mechanics or making observations about the current state of the game, including questions that aren't offers or counteroffers. 
    
    Please remember: If an utterance qualifies for "Other" but also for one of the other 4 types, it should then be considered of the other type (not of the type "Other")
    
    A game of Settlers is being played by a group of players. During the game there is this dialogue:\n
    {dialogue}
    Please consider what was said AND who said it, which can indicate a specific player's willingness to give or receive a specific resource. Reflect on how the conversation is going and the context, imagining that each utterance was written one after the other in that order. Reflect on the intentions of the players. 
    
    How could that dialogue continue?\n
    Very important: please respond with 1 possible continuation in this precise format: [type of utterance]
    """
    return prompt.replace('\t', '')


def write_results_to_file(y_true, y_pred, y_true_parsed, y_pred_parsed):
    ts = str(time.time()).split('.')[0]
    n_turns = str(CONFIG['NUM_TURNS_TO_CLASSIFY'])
    n_context = str(CONFIG['NUM_CONTEXT_SAMPLES'])
    with open('no_players_no_text_no_description_n_turns_' + n_turns + '_n_context_' + n_context +'_' + ts + '.tsv', 'w') as out_file:
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
        if len(sample) > 0:
            prompt = build_prompt(sample)
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

