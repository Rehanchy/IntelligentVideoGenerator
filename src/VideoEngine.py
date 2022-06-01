from CLIP import clip
import torch
from PIL import Image
import os
import patchify
import pickle as pkl
import cv2
import zipfile

class VideoEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None
        self.image_features = None
        self.feature_map_image = []
        self.trained_num = 0

    def set_model(self, model_name):
        #online_path = "../data/rehanchy/Clip_models_" + model_name
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def add_video(self, path, batch_size=64, patch_size=640, ms_in_feature=1000):
        for video in os.listdir(path):
            print(self.trained_num)
            video = os.path.join(path, video)
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(fps)
            frame_in_feature = int(fps/(1000/ms_in_feature))
            feature_list = []
            to_encode = []
            feature_video_map = []
            frame_idx = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if frame_idx % frame_in_feature == 0:
                    frame_patch = self.image_patch(frame, patch_size)
                    #feature_list.append(frame_patch)
                    to_encode.append(frame)
                    feature_video_map.append({'frame_idx': frame_idx, 'video_path': video, 'time' : frame_idx/fps})
                if len(to_encode) >= batch_size:
                    features = self.calculate_image_features(to_encode)
                    feature_list.append(features)
                    to_encode = []
                frame_idx += 1
            if len(to_encode) > 0:
                features = self.calculate_image_features(to_encode)
                feature_list.append(features)
            feature_t = torch.cat(feature_list, dim=0)
            self.add_image_features(feature_t, feature_video_map)
            self.trained_num += 1
    
    def preprocess_forCV(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return self.preprocess(image)    
        
    def calculate_image_features(self, images):
        for i in range(len(images)):
            images[i] = self.preprocess_forCV(images[i])
        image_stack = torch.stack(images, dim=0)
        #print(image_stack.shape)
        image_t = image_stack.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_t)
        return image_features

    def add_image_features(self, features, image_map):
        assert(features.shape[0] == len(image_map))
        features /= features.norm(dim=-1, keepdim=True)
        if self.image_features is None:
            self.image_features = features
        else:
            self.image_features = torch.cat((self.image_features, features), dim=0)
        self.feature_map_image.extend(image_map)

    def image_patch(self, image, patch_size):
        step = int(patch_size / 2)
        patch = patchify.patchify(image,(patch_size,patch_size,3),step=step)
        patches = []
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                patches.append(patch[i,j,0])
        return patches

    # input is only one sentence
    def search(self, input, n = 3):
        text_input = torch.cat([clip.tokenize(input).to(self.device)])
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        consine_similarity = (100.0 * text_features @ self.image_features.T)
        # get topk
        values, indices = consine_similarity[0].topk(n*100)
        # avoid same videos
        output_videos = set()
        matches = []
        for indice_idx, similarity_idx in enumerate(indices):
            if len(matches) >= n:
                break
            match_data = self.feature_map_image[similarity_idx]
            score = float(values[indice_idx].cpu().numpy())
            video_id = '{}'.format(match_data['video_path'])
            if video_id in output_videos:
                continue
            # store video information for making videos
            full_result = {**match_data, 'score': score}
            matches.append(full_result)
            output_videos.add(video_id)
        return matches

    # input would be list of sentences
    def list_search(self, input_list, n = 3):
        match_list = []
        output_videos = set()
        for input in input_list:
            text_input = torch.cat([clip.tokenize(input).to(self.device)])
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            consine_similarity = (100.0 * text_features @ self.image_features.T)

            values, indices = consine_similarity[0].topk(n*100)
            
            matches = []

            for indice_idx, similarity_idx in enumerate(indices):
                if len(matches) >= n:
                    break
                match_data = self.feature_map_image[similarity_idx]
                score = float(values[indice_idx].cpu().numpy())
                video_id = '{}'.format(match_data['video_path'])
                if video_id in output_videos:
                    continue
                full_result = {**match_data, 'score': score}
                matches.append(full_result)
                output_videos.add(video_id)
            match_list.append(matches)
        return match_list
    
    # context aware search method
    def list_search_context_aware(self, input_list, input_title, n = 3):
        match_list = []
        output_videos = set()
        # title weight 0.1
        title_input = torch.cat([clip.tokenize(input_title).to(self.device)])
        with torch.no_grad():
            text_features = self.model.encode_text(title_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        title_consine_similarity = (10.0 * text_features @ self.image_features.T)
        i = 0
        for i in range(len(input_list)):
            # do not process begin and end
            if i == 0 or i == len(input_list)-1:
                input = input_list[i]
                text_input = torch.cat([clip.tokenize(input).to(self.device)])
                with torch.no_grad():
                    text_features = self.model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                consine_similarity = (100.0 * text_features @ self.image_features.T)
                consine_similarity += title_consine_similarity
                values, indices = consine_similarity[0].topk(n*100)
                
                matches = []

                for indice_idx, similarity_idx in enumerate(indices):
                    if len(matches) >= n:
                        break
                    match_data = self.feature_map_image[similarity_idx]
                    score = float(values[indice_idx].cpu().numpy())
                    video_id = '{}'.format(match_data['video_path'])
                    if video_id in output_videos:
                        continue
                    full_result = {**match_data, 'score': score}
                    matches.append(full_result)
                    output_videos.add(video_id)
                match_list.append(matches)
            # make context as the sentence before and after, weight 0.2
            else:
                j = i - 1
                input = input_list[j]
                text_input = torch.cat([clip.tokenize(input).to(self.device)])
                with torch.no_grad():
                    text_features = self.model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                consine_similarity = (20.0 * text_features @ self.image_features.T) 
                j = i
                input = input_list[j]
                text_input = torch.cat([clip.tokenize(input).to(self.device)])
                with torch.no_grad():
                    text_features = self.model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                consine_similarity += (100.0 * text_features @ self.image_features.T)
                j = i + 1
                input = input_list[j]
                text_input = torch.cat([clip.tokenize(input).to(self.device)])
                with torch.no_grad():
                    text_features = self.model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                consine_similarity += (20.0 * text_features @ self.image_features.T)
                consine_similarity += title_consine_similarity
                values, indices = consine_similarity[0].topk(n*100)
                matches = []

                for indice_idx, similarity_idx in enumerate(indices):
                    if len(matches) >= n:
                        break
                    match_data = self.feature_map_image[similarity_idx]
                    score = float(values[indice_idx].cpu().numpy())
                    video_id = '{}'.format(match_data['video_path'])
                    if video_id in output_videos:
                        continue
                    full_result = {**match_data, 'score': score}
                    matches.append(full_result)
                    output_videos.add(video_id)
                match_list.append(matches)
        return match_list

    def save(self, path):
        feature_dir = self.model_name+"-features.pt"
        feature_dir = os.path.join(path, feature_dir)
        map_dir = self.model_name+"-feature_map.pkl"
        map_dir = os.path.join(path, map_dir)
        torch.save(self.image_features, feature_dir)
        pkl.dump(self.feature_map_image, open(map_dir, 'wb'))

    def load(self, path, model_name):
        self.set_model(model_name)
        if self.model_name == "ViT-B/32":
            model_name = "ViT-B-32"
        feature_dir = model_name+"-features.pt"
        feature_dir = os.path.join(path, feature_dir)
        map_dir = self.model_name+"-feature_map.pkl"
        map_dir = os.path.join(path, map_dir)
        self.image_features = torch.load(feature_dir)
        self.feature_map_image = pkl.load(open(map_dir, 'rb'))
    
    # make video for one sentence
    def MakeVideo(self, results, output_path, input, fps=30):
        print("producing Videos")
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
        for result in results:
            frames = 120
            video_path = result['video_path']
            frame_idx = result['frame_idx']
            cap = cv2.VideoCapture(video_path)
            myfps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            for i in range(frames):
                ret, frame = cap.read()
                if ret == False:
                    break
                frame = cv2.resize(frame, (1920, 1080))
                cv2.putText(frame, input, (100, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                video_writer.write(frame)
        video_writer.release()
    
    # make video for list of sentences
    def list_MakeVideo(self, results, output_path, input_list, fps=30, video_paths = None):
        print("producing Videos")
        id = 0
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
        for result in results:
            for part_result in result:
                frames = 120
                if video_paths is not None:
                    video_path = part_result['video_path']
                    useless, val = os.path.split(video_path)
                    video_path = video_paths +"/"+ val
                else:
                    video_path = part_result['video_path']
                #print(video_path)
                frame_idx = part_result['frame_idx']
                cap = cv2.VideoCapture(video_path)
                myfps = cap.get(cv2.CAP_PROP_FPS)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                for i in range(frames):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    frame = cv2.resize(frame, (1920, 1080))
                    font_size = 1.5
                    if len(input_list[id])>50:
                        font_size = 1
                    if len(input_list[id])>100:
                        font_size = 0.75
                    cv2.putText(frame, input_list[id], (100, 900), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 4)
                    video_writer.write(frame)
            id += 1
        video_writer.release()

    def list_getVideoFromZip(self, results, zip_path):
        video_list = []
        for result in results:
            for videodir in result:
                val = videodir['video_path']
                useless, val = os.path.split(val)
                video_list.append(val)
        results_path = "../output"
        path_head = "videos"
        with zipfile.ZipFile(zip_path, "r") as zfile:
            for video in video_list:
                video = path_head + "/" + video
                zfile.extract(video, path=results_path)

    def remove_output(self):
        output_path = "../output/videos"
        for img in os.listdir(output_path):
            os.remove(os.path.join(output_path, img))