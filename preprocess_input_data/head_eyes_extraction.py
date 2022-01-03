import cv2
import face_alignment
import numpy as np
import collections
import math

# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}
#you can change device to 'gpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

#second check : using the 3D technique of bulat et al article, detect eyes, else return zeros
def bulat_al(imHead,fa=fa):#second check
    try :
            preds = fa.get_landmarks(imHead)[-1]#predict
          # 2D-Plot
            pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
            pred_types = {
                    'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                    'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                    }
            centers=[]
            Avr_eye=24 #24 represents average eye length for adults, we use this to set the scale

            for pred_type in pred_types.values():
                x=preds[pred_type.slice, 0]
                y=preds[pred_type.slice, 1]
                centroid = (sum(x) / len(x), sum(y) / len(y)) #get centroid
                centers.append(centroid) #append
            dist = math.hypot(centers[0][0]-centers[1][0],centers[0][1]-centers[1][1]) #distance on image
            dist_reel= np.divide(dist*24,x.max()-x.min() )#real distance
            if (77 >dist_reel>51):  #normal distance between pupils is between 51 and 77
                centers = [(int(element[0]), int(element[1])) for element in centers]
                imEye_r=imHead[centers[0][0]-7:centers[0][0]+7,centers[0][1]-7:centers[0][1]+7,:]
                imEye_l=imHead[centers[1][0]-7:centers[1][0]+7,centers[1][1]-7:centers[1][1]+7,:]
                fl= imEye_l.flatten()
                fr= imEye_r.flatten()
                imEye_r = cv2.resize(imEye_r,(60,36))
                imEye_l = cv2.resize(imEye_l,(60,36))
            else :
                blank_image = np.zeros((36,60,3), np.uint8)
                imEye_l = blank_image
                imEye_r = blank_image
    except :
            blank_image = np.zeros((36,60,3), np.uint8)
            imEye_l = blank_image
            imEye_r = blank_image
    return imEye_l,imEye_r

def eye_head_extractor(img, fa):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = []
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face.append(img[y:y+h, x:x+w])

    eye_left, eye_right = bulat_al(face[0], fa)
    
    return face, eye_left, eye_right

