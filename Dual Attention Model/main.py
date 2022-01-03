import pickle
from DAM import DAM

gaze_estimations = pickle.load( open( "data/prediction_gazefollow.pkl", "rb" ) )
heads_poses = pickle.load( open( "data/head_poses.pkl", "rb" ) )
depth_maps_imgs = pickle.load( open( "data/depth_estimation_maps_gaze_follow.pkl", "rb" ) )
heads = pickle.load( open( "data/imgs_heads_gaze_follow.pkl", "rb" ) )
image_origin = pickle.load( open( "data/imgs.pkl", "rb" ) )

gz = gaze_estimations[:,2]

heat_maps = DAM(depth_maps_imgs, heads, gz, heads_poses, image_origin, alpha=6)