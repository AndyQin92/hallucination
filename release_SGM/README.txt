# SGM
This is the implementation of our WACV 2020 work entitled as "Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval". For research use only, commercial use is not allowed.
Version 2.0, Copyright(c) March, 2020
Sijin Wang, Ruiping Wang, Ziwei Yao, Shiguang Shan, Xilin Chen.
All Rights Reserved.


# Acknowledgment
Some part of our implementation is built on top of the opened source code of SCAN[1], whose code is available at: https://github.com/kuanghuei/SCAN.


# Citation
If you find this implementation or the analysis conducted in our report helpful, please consider citing our paper: Sijin Wang, Ruiping Wang, Ziwei Yao, Shiguang Shan, Xilin Chen. Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval. IEEE  Winter Conference on Applications of Computer Vision 2020. 


# Prerequisites
	Python 2.7
	Pytorch >= 1.0
	TensorBoard
	h5py
	Punkt Sentence Tokenizer:
		import nltk
		nltk.download()
		> d punkt    


# Datasets
For downloading pre-trained models and data, please email the corresponding author Dr. Ruiping Wang (wangruiping@ict.ac.cn).

	* Data Splits
	You can download our data splits for MS COCO[2] and Flickr30K[3].
	There are visual objects and relationships information in "flickr30k_train/val/test_image_sg_by_NeuralMotifs_36_25.json" and "NM_vgg_mscoco_train/val/test2014_image_sg_by_NeuralMotifsVGG_36_25_2.json", which we obtained by NeuralMotifs[4]. The "mscoco_train/val/test2014_image_sg_by_MSDN_36_25.json" is obtained by MSDN[5].

	There are textual tuples extracted by SPICE[6] in "*_caps_with_rel.json". We got these tuples by modifying the public source code of SPICE[6] to get its intermediate output("https://github.com/peteanderson80/SPICE"). You can find the code and instructions in "tools" dir.

	If you find the dataset and code are helpful, please cite NeuralMotifs[4], MSDN[5] and SPICE[6].

	* Visual Features
	We use bounding boxes in our data splits as proposals to extract visual features by the detector of bottom-up-attention[7].
	You can find the code and instructions in "tools" dir.
	If you find the dataset and code are helpful, please cite bottom-up-attention[7].

	* Pre-trained Checkpoints and the Glove[8] File are also available. 



# Training
When you have got the data splits and visual features, you can use the parameter setting in our paper or the shell scripts to train the models.


# Evaluation
You can use the eval.py to evaluate the trained models. Set the "fold5", "model_name" and the parameters of "evaluation.evalrank()". Note when you evaluate the model trained on Flickr30k, please set the "fold5" to False.


# Contact
If you have any problem about our code, feel free to contact sijin.wang@vipl.ict.ac.cn or wangruiping@ict.ac.cn

# Reference
[1] Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching. In Proceedings of the European Conference on Computer Vision, pages 201–216, 2018.
[2] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollar. Microsoft coco: Common objects in context. In Proceedings of the European Conference on Computer Vision, pages 740–755. Springer, 2014.
[3] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. In Transactions of the Association for Computational Linguistics, volume 2, pages 67–78. MIT Press, 2014.
[4] Rowan Zellers, Mark Yatskar, Sam Thomson, and Yejin Choi. Neural motifs: Scene graph parsing with global context. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5831–5840, 2018.
[5] Yikang Li, Wanli Ouyang, Bolei Zhou, Kun Wang, and Xiaogang Wang. Scene graph generation from objects, phrases and region captions. In International Conference in Computer Vision, pages 1261–1270, 2017.
[6] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In Proceedings of the European Conference on Computer Vision, pages 382–398. Springer, 2016.
[7] Peter Anderson, Xiaodong He, Chris Buehler, and Damien Teney. Bottom-up and top-down attention for image captioning and visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6077–6086, 2018.
[8] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014. https://nlp.stanford.edu/projects/glove/