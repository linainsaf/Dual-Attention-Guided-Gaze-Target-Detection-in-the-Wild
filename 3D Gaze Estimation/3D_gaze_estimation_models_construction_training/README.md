# 3D gaze estimation part :  models constructions

=======
We used head images of the gaze360 to train our models. For memory constrains we reduced the dataset.

link to reduced gaze360 dataset : https://drive.google.com/file/d/1ji6ktteZpMcvBHoiTGas1qA93EP7PlD7/view?usp=sharing

This folder includes the notebook codes and python function that allowed us to construct our 3D gaze estimation models. 

"gaze360gazeEstimation_h_features_extraction.ipynb" file contains steps of h estimation, eyes detection and features extraction.
"MLP_gaze_Estimation.ipynb" file contains the construction of the Gaze estimation MLP network that will give us the output of the 3D gaze estimation gx,gy,gz. 

All models of this part were trained using gaze360 dataset. 

A seperate code "FaceAlignment_for_eyes_detection.ipynb" was created to extract eyes from any head image using the bulat et al method.

Links to the the intermediate data files for the chosen 5000 image from gaze360 dataset : 

* h list : https://drive.google.com/file/d/1-I8cdSM0HRbns2xjiAgz1UjT5cchvduT/view?usp=sharing
* heads images : https://drive.google.com/file/d/1-A9pyyscdclW7d1i__N1oZGfgxOT3dVu/view?usp=sharing
* left eye images : https://drive.google.com/file/d/1Cjn_xsU1Qt7wGSkIKYeJpXn0INgwyNoB/view?usp=sharing
* right eye images : https://drive.google.com/file/d/1-2R2F5UqjvspcE3MEPfJ34TMp2qLxhxP/view?usp=sharing
* left eye features : https://drive.google.com/file/d/1NGw8f_XqjRydYTwOSGRNL8P8n7yYaioL/view?usp=sharing
* right eye features : https://drive.google.com/file/d/1-9xs-k44KVWffxtdC5Un4SHPAe55Hlet/view?usp=sharing
* gaze list : https://drive.google.com/file/d/1-I3BSzdODbh8eDDdKS_lo9pKNdH2Ikcg/view?usp=sharing

Links to train models : 

* Head pose extractor : https://drive.google.com/file/d/1_qmQrblnvX2KJOuAVQH82e97m92XXrU3/view?usp=sharing
* 3D gaze estimation MLP : https://drive.google.com/file/d/19t9XhsP4EENQ7EiGbiY64tt2elFAAkjy/view?usp=sharing
