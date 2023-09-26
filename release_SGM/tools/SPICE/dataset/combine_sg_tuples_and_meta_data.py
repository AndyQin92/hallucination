import os
import json

""" add the sg tuples to the meta data """


def combine(meta_file,tuple_file,dist_file):
	with open(tuple_file, 'r') as f:
		tuple_data = json.load(f)
	with open(meta_file, 'r') as f:
		meta_data = json.load(f)

	imgid_to_sgs = dict()
	for sg in tuple_data:
		#print sg
		imgid = sg['imageid']
		if imgid not in imgid_to_sgs:
			imgid_to_sgs[imgid] = []
		imgid_to_sgs[imgid].append({"sentid": sg['sentid'], "scene": sg['scene']})

	for img in meta_data['images']:
		imgid = img["imageid"]
		sgs = imgid_to_sgs[imgid]
		img['tuples'] = sgs


	dist_data = meta_data
	with open(dist_file, 'w') as f:
		json.dump(dist_data, f) 


if __name__ == '__main__':
	combine('example_input.json', 'example_output.json', 'example_combine.json')
