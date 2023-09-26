from vocab import Vocabulary
import evaluation
import json
fold5 =False
model_name ='flickr30k_BUA_sg_with_sg_label_fusion_with_spice_sg_lstm_mode2_20190411_checkpoint_21'
t2i_results, i2t_results = evaluation.evalrank("runs/flickr30k_BUA_sg_with_sg_label_fusion_with_spice_sg_lstm_mode2_20190411/checkpoint_21.pth.tar", data_path="../dataset/Flickr30k", split="test", fold5=fold5)
dist_file = model_name + '_result_fold5_%s.json'%str(fold5)

results = dict()
results['t2i'] = t2i_results
results['i2t'] = i2t_results

with open(dist_file, 'w') as f:
	json.dump(results, f)