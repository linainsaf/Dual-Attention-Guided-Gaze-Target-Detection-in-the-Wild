# Dual-Attention-Guided-Gaze-Target-Detection-in-the-Wild
projet MLA M2
<<<<<<< HEAD
link to the reduced dataset of gaze 360: https://drive.google.com/file/d/1ji6ktteZpMcvBHoiTGas1qA93EP7PlD7/view?usp=sharing 
=======
link to reduced gaze360 dataset : https://drive.google.com/file/d/1ji6ktteZpMcvBHoiTGas1qA93EP7PlD7/view?usp=sharing
>>>>>>> 33d6f1ab75b4b53a4432b19bff730d3d5619a278

This folder includes the notebook codes and python function that allowed us to construct our 3D gaze estimation models. 

"gaze360gazeEstimation_h_features_extraction.ipynb" file contains steps of h estimation, eyes detection and features extraction.
"MLP_gaze_Estimation.ipynb" file contains the construction of the Gaze estimation MLP network that will give us the output of the 3D gaze estimation gx,gy,gz. 

All models of this part were trained using gaze360 dataset. 

A seperate code "FaceAlignment_for_eyes_detection.ipynb" was created to extract eyes from any head image using the bulat et al method.
