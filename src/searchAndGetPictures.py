import PictureEngine

# specify text and image path here
zip_file_path = "../images.zip"
PictureEngine = PictureEngine.PictureEngine()
PictureEngine.load("../model/trained/RN50x4",model_name="RN50x4")
image_text_file_path = "../image_annotations.json"

# input
input_text = "Two people walking in the woods"
#input_text = "Children are playing together"
results = PictureEngine.search(input_text, n = 5)
PictureEngine.getPicturesFromZip(input_text, results, image_text_file_path, zip_file_path)
# load pictures from dir by uncommenting below
#PictureEngine.getPictures(input_text, results, image_text_file_path, pictures_path=)

# you could remove output by uncommenting the sentence below
# PictureEngine.remove_output()

    
    