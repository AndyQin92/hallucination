# bottom-up-attention
We use the pre-trained detector in bottom-up-attention[1] to extract our object visual features and relationship visual features. We modified part of the code of bottom-up-attention. To use our code, you need to download the release code from https://github.com/peteanderson80/bottom-up-attention/tree/29167f36d9059e6645c05baa3c70e3a040953e85 first. And then install the experimental environment according to their instructions, and download their pre-trained model which is named "resnet101_faster_rcnn_final.caffemodel". Then replace part of their code with ours in this "bottom-up-attention" dir. Finally, you can use the example commands showed in "extract_flickr30k_train.sh" and "extract_flickr30k_rel_train.sh" to extract the object features and relationship features respectively. As the .h5 file which contains features of all images is too large to load in the memory during training. So we save features of each image in an individual .npy file using fearures/creat_npys.py.
If you find the dataset and code are helpful, please cite bottom-up-attention[1].

# SPICE
We use the SPICE[2] to extract tuples from text sentences. We modified part of the code of SPICE[2] (from https://github.com/peteanderson80/SPICE).
First, you need to install java 1.8.0+. Then "cd ./SPICE" and use the command "mvn clean verify" to build the source code. Then "cd ./target" and use this command "java -Xmx8G -jar spice-*.jar ../dataset/example_input.json -out ../dataset/example_output.json -mode test" to extract tuples from the "example_input.json". Finally, use ./dataset/combine_sg_tuples_and_meta_data.py to combine the tuples and original sentences.
If you find the dataset and code are helpful, please cite SPICE[2].



[1] Peter Anderson, Xiaodong He, Chris Buehler, and Damien Teney. Bottom-up and top-down attention for image captioning and visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6077–6086, 2018.

[2] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In Proceedings of the European Conference on Computer Vision, pages 382–398. Springer, 2016.