import VideoEngine
VideoEngine = VideoEngine.VideoEngine()
VideoEngine.load("../model/video_trained/RN50x4",model_name="RN50x4")
video_path = "F:/data/videos.zip"
input_title = "Dumplings. "
#input_text = "A man and a woman chatting, they are in the city. Exercising benifits us a lot, see this man. Peoples are hanging out in the woods. The blue sky is a beautiful sence for us. The national flag are waving in the air. The spacecraft are flying in the space."
input_text = "Dumpling is a traditional chinese food. On the lunar new year' s day, most families make a lot of delicious dumplings. To make them, follow this easy process. The first step is to mix the flour with water. when the dough is ready, we can begin to make dumpling wrappers."
input_text = input_text + "we use a rolling pole to roll the dough into small, thin and round pieces so that they will be easy to cook. when the wrappers are done, it' s time to prepare the filling. Usually, we use meat such as beef or mutton, and some vegetables such as cabbage or carrots for filling.You may actually choose whatever you like to put into it. we must chop those things into small pieces.Then put some salt, oil and other condiments into it and stir it evenly. When all these preparations are done, we can start making dumplings. First put a spoonful of filling in the center of the wrapper. Then stick the two opposite sides together and then the rest. Don' t just cover the filling with the wrappers. The best shape of dumplings is that of a ship, because when they are boiling in the pan, they look like ships sailing in the sea. The smell can make your mouth water."
# devide sentences
input_list = str.split(input_text, ".")
for input in input_list:
    if len(input) < 3:
        input_list.remove(input)
results = VideoEngine.list_search_context_aware(input_list,input_title,1)
# get video from zips, if you do not need it from zipfile, comment this sentence
VideoEngine.list_getVideoFromZip(results, video_path)
VideoEngine.list_MakeVideo(results, "./productVideo.mp4", input_list, video_paths="../output/videos")
results = VideoEngine.list_search(input_list,1)
VideoEngine.list_getVideoFromZip(results, video_path)
VideoEngine.list_MakeVideo(results, "./productVideo1.mp4", input_list, video_paths="../output/videos")
#input_text = "Amy is the most beautiful girl in my class. She has long and black hair, a round face. She looks lovely and beautiful when she smiles. She is so outgoing that she is popular among classmates. She is so energetic that she is likely to take part in various activities. Therefore, she makes many good friends. Besides, she is always ready to help others. Many classmates get help from her both in study and life. She works hard in study and she does well in all subjects, so that when we have questions on study, we would like to ask her. I think her personalities make her beautiful."
