# Multimodal incremental learning from human interaction.

## Prerequisites:

* Data input: all the videos should be organized and named in folders by users and actions. Each action has a folder for each sensor. For RGB-D the files will be separated in two folderd, one for RGB data and another one for the depth map. For now, the speech is just a txt file.
	
* Libraries: Sklearn, Scipy, Numpy, OpenCV 2.8.
	
* This code needs to have a base model to obtain the skeleton, adapted from the work of Zhe Cao [1]. It is saved with Caffe framework format. To obtain the corresponding caffemodel you need to launch the script in Interaction/model/get_model.sh.
	
[1] *Realtime Multi-Person Pose Estimation*. By Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

## Launch instructions:

To launch the system: 

	python Test.py userX actionX

This will create a .gpz file with all the data compressed, so you can launch the same video multiple times in a faster way.
First execution will create a .pkl with the learning model created. Afterwards, it will update the pkl file.
