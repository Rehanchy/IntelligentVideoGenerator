import numpy as np
from CLIP import clip
import torch
from PIL import Image
import os
import patchify
import pickle as pkl
import zipfile
import json
import cv2
from simtext import similarity

class PictureEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None
        self.image_features = None
        self.feature_map_image = []
        self.trained_num = 0

    # set model before adding images
    def set_model(self, model_name):
        #online_path = "../data/rehanchy/Clip_models_" + model_name
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def add_image(self, path, batch_size=32):
        feature_list = []
        image_to_encode = []
        feature_image_map = []
        for img in os.listdir(path):
            img = os.path.join(path, img)
            self.trained_num += 1
            if(self.trained_num % 1000 == 0):
                print("trained {} images".format(self.trained_num))
            imgdir = img
            img = np.asarray(Image.open(img))
            feature_detail = {'image_path': imgdir}
            feature_image_map.append(feature_detail)
            image_to_encode.append(img)
            if len(image_to_encode) >= batch_size:
                feature_list.append(self.calculate_image_features(image_to_encode))
                image_to_encode = []
        if len(image_to_encode) > 0:
            feature_list.append(self.calculate_image_features(image_to_encode))
        feature_t = torch.cat(feature_list, dim=0)
        self.add_image_features(feature_t, feature_image_map)
        
    # use patchify method    
    def add_image_with_patch(self, path, batch_size=32, patch_size=640):
        feature_list = []
        image_to_encode = []
        feature_image_map = []
        for img in os.listdir(path):
            img = os.path.join(path, img)
            self.trained_num += 1
            if(self.trained_num % 1000 == 0):
                print("trained {} images".format(self.trained_num))
            imgdir = img
            img = np.asarray(Image.open(img))
            if(img.shape[0]<640 or img.shape[1]<640):
                patches = [img]
            else:
                patches = self.image_patch(img, patch_size) + [img]
            for idx, patch in enumerate(patches):
                feature_detail = {'image_path': imgdir}
                feature_image_map.append(feature_detail)
                image_to_encode.append(patch)
            if len(image_to_encode) >= batch_size:
                feature_list.append(self.calculate_image_features(image_to_encode))
                image_to_encode = []
        if len(image_to_encode) > 0:
            feature_list.append(self.calculate_image_features(image_to_encode))
        feature_t = torch.cat(feature_list, dim=0)
        self.add_image_features(feature_t, feature_image_map)

    # get representations of pictures through CLIP
    def calculate_image_features(self, images):
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i])
            images[i] = self.preprocess(images[i])
        image_stack = torch.stack(images, dim=0).cuda()
        image_temp_f = image_stack.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_temp_f)
        return image_features

    # concat features and maps
    def add_image_features(self, features, image_map):
        features /= features.norm(dim=-1, keepdim=True)
        if self.image_features is None:
            self.image_features = features
        else:
            self.image_features = torch.cat((self.image_features, features), dim=0).cuda()
        self.feature_map_image.extend(image_map)

    # patchify
    def image_patch(self, image, patch_size):
        step = int(patch_size / 2)
        patch = patchify.patchify(image,(patch_size,patch_size,3),step=step)
        patches = []
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                patches.append(patch[i,j,0])
        return patches

    # search similar images
    def search(self, input, n = 10):
        text_input = torch.cat([clip.tokenize(input).to(self.device)]).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        consine_similarity = (100.0 * text_features @ self.image_features.T)
        values, indices = consine_similarity[0].topk(n*100)
        output_images = set()
        matches = []
        for indice_idx, similarity_idx in enumerate(indices):
            if len(matches) >= n:
                break
            match_data = self.feature_map_image[similarity_idx]
            score = float(values[indice_idx].cpu().numpy())
            img_id = '{}'.format(match_data['image_path'])
            if img_id in output_images:
                continue
            # store image information
            full_result = {**match_data, 'score': score}
            matches.append(full_result)
            output_images.add(img_id)
        return matches

    # save features and its map
    def save(self, path):
        if self.model_name == "ViT-B/32":
            model_name = "ViT-B-32"
        feature_dir = model_name+"-features.pt"
        feature_dir = os.path.join(path, feature_dir)
        map_dir = model_name+"-feature_map.pkl"
        map_dir = os.path.join(path, map_dir)
        torch.save(self.image_features, feature_dir)
        pkl.dump(self.feature_map_image, open(map_dir, 'wb'))

    # load from saved features and its map
    def load(self, path, model_name):
        self.set_model(model_name)
        if self.model_name == "ViT-B/32":
            model_name = "ViT-B-32"
        feature_dir = model_name+"-features.pt"
        feature_dir = os.path.join(path, feature_dir)
        map_dir = model_name+"-feature_map.pkl"
        map_dir = os.path.join(path, map_dir)
        self.image_features = torch.load(feature_dir)
        self.feature_map_image = pkl.load(open(map_dir, 'rb'))

    # get images
    def getPictures(self, input_text, results, text_file, pictures_path):
        img_results = []
        Myscore = []
        # get image names
        for imgdir in results:
            val = imgdir['image_path']
            Myscore.append(imgdir['score'])
            useless, val = os.path.split(val)
            img_results.append(val)
        # show images from results
        i = 0
        for img in img_results:
            with open(text_file,'r',encoding="utf-8") as fr:
                FileDict = json.load(fr)
                for imgdict in FileDict:
                    caption, img_name = imgdict.items()
                    if img_name[1] == img:
                        Word_Score = 0
                        sim = similarity()
                        Word_Score = sim.compute(caption[1],input_text)['Sim_Cosine']
                        img = os.path.join(pictures_path, img)
                        img = cv2.imread(img)
                        cv2.resize(img, (1920,1080))
                        place = (20,20)
                        cv2.putText(img,  caption[1], place, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                        place = (20,40)
                        cv2.putText(img,  input_text, place, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        place = (20,60)
                        cv2.putText(img,  "Score: " + str(Myscore[i]) + " WordScore: " + str(Word_Score), place, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                        cv2.imshow("img", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        i += 1
                        break
    # get images directly from zip file
    def getPicturesFromZip(self, input_text, results, text_file, zipfile_path):
        img_results = []
        Myscore = []
        # get image names
        for imgdir in results:
            val = imgdir['image_path']
            Myscore.append(imgdir['score'])
            useless, val = os.path.split(val)
            img_results.append(val)
        # get image from zip
        output_path = "../output"
        with zipfile.ZipFile(zipfile_path, "r") as zfile:
            for img in img_results:
                zfile.extract(img, path=output_path)
        # show images
        i = 0
        for img in img_results:
            with open(text_file,'r',encoding="utf-8") as fr:
                FileDict = json.load(fr)
                for imgdict in FileDict:
                    caption, img_name = imgdict.items()
                    if img_name[1] == img:
                        Word_Score = 0
                        sim = similarity()
                        Word_Score = sim.compute(caption[1],input_text)['Sim_Cosine']
                        img = os.path.join(output_path, img)
                        img = cv2.imread(img)
                        cv2.resize(img, (1920,1080))
                        place = (20,20)
                        cv2.putText(img,  caption[1], place, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                        place = (20,40)
                        cv2.putText(img,  input_text, place, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        place = (20,60)
                        cv2.putText(img,  "Score: " + str(Myscore[i]) + " WordScore: " + str(Word_Score), place, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                        cv2.imshow("img", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        i += 1
                        break
    def remove_output(self):
        output_path = "../output"
        for img in os.listdir(output_path):
            os.remove(os.path.join(output_path, img))