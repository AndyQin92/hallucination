import h5py
import numpy as np
import os



pre='/data/image_text_matching_using_scene_graph/dataset/Flickr8k/'
from_file = 'NM_vgg_flickr8k_test_image_rel_features_by_BUA_25.h5'
dist_dir = pre+'npys/NM_vgg_flickr8k_test_image_rel_features_by_BUA_25'




f = h5py.File(from_file, 'r')
data = f['data'][:]
print(data.shape)

f.close()

for i in range(data.shape[0]):
	if i%1000==0:
		print i
	np.save(os.path.join(dist_dir, str(i)+'.npy'), data[i])
	

