import VideoEngine

# this file is used for preload video features
VideoEngine = VideoEngine.VideoEngine()
VideoEngine.set_model("RN50x4")
VideoEngine.add_video("../data/Zeeeee/USTC_MSRA_2022_Proj8_data/Proj8/videos/videos")
VideoEngine.save("../model")
