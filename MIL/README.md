# Multimodal incremental learning from human interaction.

## Prerequisites:

* Data input: all the videos should be organized and named in folders by users and actions. Each action has a folder for each sensor. For RGB-D the files will be separated in two folderd, one for RGB data and another one for the depth map. For now, the speech is just a txt file. 
	
* Libraries: Sklearn, Scipy, Numpy, OpenCV 2.4.8.2. It also uses a virtual environment for Tensorflow and python3 to be able to run MaskRCNN. This is all under tfenv folder. To install everything launch the script install.sh that populates the tfenv folder with the Matterport version of MaskRCNN[2].
	
* This code needs to have a base model to obtain the skeleton, adapted from the work of Zhe Cao [1] and CaffeNet. To obtain the corresponding all the models you need to launch the script in get_model.sh.

Example of Data input:

	├── user1
	│   ├── point_1
	│   │   ├── audio
	│   │   ├── k1
	│   │   │   ├── Depth
	│   │   │   └── RGB
	│   │   ├── k2
	│   │   │   ├── Depth
	│   │   │   └── RGB
	│   │   └── usb
	│   │       ├── Depth
	│   │       └── RGB
	......

[1] *Realtime Multi-Person Pose Estimation*. By Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[2] https://github.com/matterport/Mask_RCNN

## Launch instructions:

To launch the system: 

	python Main.py userX actionX

If you choose to train this will create a .gpz file with all the data compressed, so you can launch the same video multiple times in a faster way.
First execution will create a .pkl with the learning model created. Afterwards, it will update the pkl file.
If you choose to test this will create multiple images files with the label assigned in the classification step.
