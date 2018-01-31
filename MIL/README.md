# Multimodal incremental learning from human interaction.

-Prerequisite:

	-Data input: Each video will be separate in users and actions. Each action has a folder for each sensor. For RGB-D the files will be separate in two folder: one for RGB and one for the depth map. The speech will be set for the moment as a txt file.
	-Libraries: Sklearn, Scipy, Numpy, OpenCV 2.8.
	-The model uses for obtaining the skeleton is adapted from the work of Zhe Cao[1]. To obtain the caffemodel you need to launch the script in Interaction/model/get_model.sh.
	
[1]Realtime Multi-Person Pose Estimation. By Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
-Output:

	-To launch the system: python Test.py userX actionX --> It will create a .gpz file with all the data compress so you can launch the same video multiple times faster. First execution will create a .pkl with the learning model created. Afterwards, it will update the pkl file.
