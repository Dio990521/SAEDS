import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fairseq.models.lstm import LSTMModel
import jieba
import pickle as pkl
checkpoint_path = "/local/ssd_1/stc/stc_clm/"

stc = LSTMModel.from_pretrained(checkpoint_path, checkpoint_file='checkpoint_best.pt',
                                data_name_or_path=checkpoint_path+'stc_ori', beam=5)
stc.eval()

f = open("/local/ssd_1/chengzhang/SA_dialog/dialogue/datas/stc_dict.pkl", 'wb')

pkl.dump(stc.tgt_dict.indices,f)
f.close()
#print(type(stc.tgt_dict.indices), len(stc.tgt_dict.indices), stc.tgt_dict.indices)
#input_sent = 
#input_sent = ' '.join(jieba.cut(''.join(input_sent.split()), cut_all=False))

#target_sent = 
#target_sent = ' '.join(jieba.cut(''.join(target_sent.split()), cut_all=False))

#criterion = stc.task.build_criterion(stc.args)
#criterion.ret_dist = True
#loss = stc.get_clm(input_sent, target_sent, criterion)
#print(stc.translate(input_sent))
#print(loss)



# target_sent_id = [trg_dict[x] if x in trg_dict.indices else trg_dict.unk_index for x in target_sent.split()]