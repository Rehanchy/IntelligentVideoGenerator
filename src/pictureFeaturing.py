import PictureEngine

# this file is used for preload image features
PictureEngine = PictureEngine.PictureEngine()
PictureEngine.set_model("RN50x4")
PictureEngine.add_image("../data/Zeeeee/USTC_MSRA_2022_Proj8_data/Proj8/images")
PictureEngine.save("../output")
# save the features in output