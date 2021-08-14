# encoding:utf-8
import os, math
from copy import copy
import time, random
import numpy as np
import argparse
import pickle as pkl
from utils.utils_sa import *
import data
from emo_inference import inference_emotion
from torch.utils.data import Dataset, DataLoader
from seq2seq_inference import *
import math

def simulatedAnnealing_dialog(option):
    batch_size = option.batch_size
    dataclass = data.Data(option) 
    use_data = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    temperatures =  option.C * (1.0 / 100) * np.array(list(range(option.sample_time + 1, 1, -1)))
    option.temperatures = temperatures
    stc_source = []
    stc_emotion = []
    with open(option.source_data_path) as f:
        for line in f.readlines():
            stc_source.append(line.strip())  
    with open(option.emotion_data_path) as f:
        for line in f.readlines():
            stc_emotion.append(line.strip()) 
    pointer = 0
    all_k_buffer_handler = [BufferHandler(os.path.join(option.this_expsdir, option.save_path))]

    for sen_id in range(int(use_data.length/batch_size)):
        input, sequence_length, _ = use_data(batch_size, sen_id)
        sources = stc_source[pointer:pointer+batch_size]
        emotions = stc_emotion[pointer:pointer+batch_size]
        pointer += batch_size
        print('------------------------')      
        for i in range(batch_size):
            print(sources[i], 'target emotion: ' + str(emotions[0]))
            print(' '.join(id2sen(input[i])))
            print(input[i])

        for k in range(option.N_repeat):
            sens, final_emo_probs = sa_dialog(input, sequence_length, sources, id2sen, option, batch_size, emotions)
            for i in range(batch_size):
                sen = ' '.join(id2sen(sens[i]))
                
                all_k_buffer_handler[k].appendtext(sen.replace('<s>','').replace('</s>','').strip())

    # Close all k files
    for k in range(option.N_repeat):
        all_k_buffer_handler[k].close()

def sa_dialog(input, sequence_length, sources, id2sen, option, batch_size, emotions):
    pos = 0
    original_text = getOriginalText(input, id2sen)
    emotion_old, = inference_emotion(original_text, emotions[0], batch_size)
    probs_old, _ = seq2seq_model(original_text, input, sources, sequence_length, id2sen)
    for iter in range(option.sample_time):
        temperature = option.temperatures[iter]
        ind = pos % (sequence_length[0] - 1)
        action = choose_action(option.action_prob)

        if action == 0: # word replacement (action: 0)
            
            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)
            input_forward_text = getOriginalText(input_forward, id2sen)
            input_backward_text = getOriginalText(input_backward, id2sen)
            _ , prob_forward = seq2seq_model(input_forward_text, input_forward, sources, sequence_length_forward, id2sen)
            _ , prob_backward = seq2seq_model(input_backward_text, input_backward, sources, sequence_length_backward, id2sen)
            
            for i in range(batch_size):
                prob_old = np.power(probs_old[i].item(), 1.0 / sequence_length[i]) * np.power(emotion_old[i].item(), option.emo_weight)
                prob_forward = prob_forward[i, ind%(sequence_length[i]-1),:]
                prob_backward = prob_backward[i, sequence_length[i]-1-ind%(sequence_length[i]-1)-1,:]
                prob_mul = prob_forward * prob_backward
                for i in range(len(prob_mul)):
                    prob_mul[i].item() = 1
                input_candidate, sequence_length_candidate = generate_candidate_input(input[i],\
                    sequence_length[i], ind, prob_mul, option.search_size, option, mode=action)
              
                # compute prob of each candidate
                input_candidate_text = getOriginalText(input_candidate, id2sen)
                prob_candidate, _ = seq2seq_model(input_candidate_text, input_candidate, sources*len(input_candidate_text), sequence_length_candidate, id2sen)
                emotion_new = inference_emotion(input_candidate_text, emotions[0], batch_size=option.search_size)
                prob_new = prob_candidate.copy()
                for j in  range(len(prob_candidate)):
                    prob_candidate[j] = np.power(prob_candidate[j].item(), 1.0 / sequence_length_candidate[j]) * np.power(emotion_new[j].item(), option.emo_weight)
                prob_candidate = np.array(prob_candidate)

                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                V_new, V_old, alphat = computeScore(prob_candidate_prob, prob_old, temperature)

                if choose_action([alphat, 1 - alphat]) == 0 and input_candidate[prob_candidate_ind][ind] < option.dict_size:
                    input_new = input_candidate[prob_candidate_ind : prob_candidate_ind + 1]
                    if np.sum(input_new[0]) == np.sum(input[i]):
                        pass
                    else:
                        input[i] = input_new
                        print('ind, action, old emotion, new emotion, vold, vnew, alpha', ind, action, emotion_old[i], emotion_new[i], V_old, V_new, alphat)
                        print('Temperature:{:3.3f}:   '.format(temperature) + ' '.join(id2sen(input[i])), sequence_length[i])
                        emotion_old[i] = emotion_new[i]
                        probs_old[i] = prob_new[prob_candidate_ind]

        elif action == 1: # word insert
            stop_insert = False
            for i in range(batch_size):
                if sequence_length[i] >= option.num_steps or ind==0:
                    pos += 1
                    stop_insert = True
                    break
            if stop_insert:
                continue
                
            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)
            input_forward_text = getOriginalText(input_forward, id2sen)
            input_backward_text = getOriginalText(input_backward, id2sen)
            _ , prob_forward = seq2seq_model(input_forward_text, input_forward, sources, sequence_length_forward, id2sen)
            _ , prob_backward = seq2seq_model(input_backward_text, input_backward, sources, sequence_length_backward, id2sen)
            
            for i in range(batch_size):
                prob_old = np.power(probs_old[i].item(), 1.0 / sequence_length[i]) * np.power(emotion_old[i].item(), option.emo_weight)
                prob_forward = prob_forward[i, ind%(sequence_length[i]-1),:]
                prob_backward = prob_backward[i, sequence_length[i]-1-ind%(sequence_length[i]-1),:]
                prob_mul = prob_forward * prob_backward
                for i in range(len(prob_mul)):
                    prob_mul[i] = 1

                input_candidate, sequence_length_candidate = generate_candidate_input(input[i],\
                    sequence_length[i], ind, prob_mul, option.search_size, option, mode=action)
                input_candidate_text = getOriginalText(input_candidate, id2sen)
                prob_candidate, _ = seq2seq_model(input_candidate_text, input_candidate, sources*len(input_candidate_text), sequence_length_candidate, id2sen)
                emotion_new = inference_emotion(input_candidate_text, emotions[0], batch_size=option.search_size)
                prob_new = prob_candidate.copy()
                for j in  range(len(prob_candidate)):
                    prob_candidate[j] = np.power(prob_candidate[j].item(), 1.0 / sequence_length_candidate[j]) * np.power(emotion_new[j].item(), option.emo_weight)
                prob_candidate = np.array(prob_candidate)

                prob_candidate_norm=normalize(prob_candidate)

                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
  
                V_new, V_old, alphat = computeScore(prob_candidate_prob, prob_old, temperature)
 
                if choose_action([alphat, 1 - alphat]) == 0 and input_candidate[prob_candidate_ind][ind] < option.dict_size:
                    input[i] = input_candidate[prob_candidate_ind : prob_candidate_ind + 1]
                    sequence_length[i] += 1
                    pos += 1
                    print('ind, action, old emotion, new emotion, vold, vnew, alpha', ind, action, emotion_old[i], emotion_new[i], V_old, V_new, alphat)
                    print('Temperature:{:3.3f}:   '.format(temperature) + ' '.join(id2sen(input[i])), sequence_length[i])
                    emotion_old[i] = emotion_new[i]
                    probs_old[i] = prob_new[prob_candidate_ind]

        elif action == 2: # word delete
            stop_delete = False
            for i in range(batch_size):
                if sequence_length[i]<=5 or ind==0:
                    pos += 1
                    stop_delete = True
                    break
            if stop_delete:
                continue
                
            for i in range(batch_size):
                prob_old = np.power(probs_old[i].item(), 1.0 / sequence_length[i]) * np.power(emotion_old[i].item(), option.emo_weight)
                input_candidate, sequence_length_candidate = generate_candidate_input(input[i],\
                    sequence_length[i], ind, None, option.search_size, option, mode=action)
                input_candidate=np.array(input_candidate)
                input_candidate_text = getOriginalText(input_candidate, id2sen)
                prob_candidate, _ = seq2seq_model(input_candidate_text, input_candidate, sources*len(input_candidate_text), sequence_length_candidate, id2sen)
                emotion_new = inference_emotion(input_candidate_text, emotions[0], batch_size=1)
                prob_new = prob_candidate.copy()
                prob_candidate[0] = np.power(prob_candidate[0].item(), 1.0 / sequence_length_candidate[0]) * np.power(emotion_new[0].item(), option.emo_weight)
                
                V_new, V_old, alphat = computeScore(prob_candidate[0], prob_old, temperature)
                
                if choose_action([alphat, 1 - alphat]) == 0 and input_candidate[0][ind] < option.dict_size:
                    input[i] = np.concatenate([input[i,:ind+1], input[i,ind+2:], input[i,:1]*0+2], axis=0) # 2 = EOS
                    sequence_length[i] -= 1
                    pos -= 1
                    print('ind, action, old emotion, new emotion, vold, vnew, alpha', ind, action, emotion_old[i], emotion_new[i], V_old, V_new, alphat)
                    print('Temperature:{:3.3f}:   '.format(temperature) + ' '.join(id2sen(input[i])), sequence_length[i])
                    emotion_old[i] = emotion_new[i]
                    probs_old[i] = prob_new[0]

        pos += 1
        
    final_emo_probs = emotion_old[0].item()
    return input, final_emo_probs

def computeScore(prob_candidate, prob_old, temperature):
    V_new = math.log(max(prob_candidate, 1e-200))
    V_old = math.log(max(prob_old, 1e-200))
    alphat = min(1, math.exp(min((V_new - V_old) / temperature, 100)))
    return V_new, V_old, alphat
        
def getOriginalText(input, id2sen):
    original_text = []
    for i in range(len(input)):
        text = ' '.join(id2sen(input[i]))
        original_text.append(text.replace('<s>','').replace('</s>','').strip())
    return original_text
    
