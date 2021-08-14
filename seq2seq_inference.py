from fairseq.models.lstm import LSTMModel
import pickle
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("loading seq2seq model...")
checkpoint_path = "/local/ssd_1/stc/nlpcc_2017_256/"

seq2seq = LSTMModel.from_pretrained(checkpoint_path, checkpoint_file='checkpoint.best_bleu_1.02.pt',
                                data_name_or_path=checkpoint_path, beam=20)
seq2seq.cuda()
seq2seq.eval()

criterion = seq2seq.task.build_criterion(seq2seq.args)
criterion.ret_dist = True
print("Done")
def seq2seq_model(inputs, inputs_idx, sources, sequence_length, id2sen):
    sequence_length = sequence_length - 1
    probs = []
    output_batch = []
    for i in range(len(inputs)):
        target_sentence = sources[i]
        output = seq2seq.get_clm(target_sentence, inputs[i], criterion)
        output_batch.append(output.cpu().data.numpy())
        prob = 1
        for j in range(sequence_length[i]-1):
            prob *= output[j][inputs_idx[i][j+1]]
        prob *= output[sequence_length[i]-1][2] # 2 = EOS
        probs.append(prob)
    return np.asarray(probs), np.asarray(output_batch)