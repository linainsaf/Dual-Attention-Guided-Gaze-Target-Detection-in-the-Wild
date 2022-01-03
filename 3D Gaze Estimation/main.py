from gaze_prediction import gaze_estimation
import pickle


heads = pickle.load( open( "data/imgs_heads_gaze_follow.pkl", "rb" ) )
eyes_l, eyes_r = pickle.load( open( "data/eyes.pkl", "rb" ) )

prediction = gaze_estimation(heads, eyes_l, eyes_r)




