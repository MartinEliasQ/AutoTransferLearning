"""
Libraries for using system or common functions
"""
from datetime import datetime
import os
import math
# import sys
import numpy as np
import pathlib

"""
Libraries for manipulating images
"""
import cv2
import dlib

"""
Other libraries
"""
# import imgaug as ia
# from imgaug import augmenters as iaa
# from skimage import io
import shutil
from sklearn.cross_validation import train_test_split
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa


class Preprocessing(object):
    """
    A preprocessing dataset images
    """

    def __init__(self, args: {}):
        self.data_folder = args['data_folder']
        self.faces_folder = args['faces_folder']
        self.video_folder = args['video_folder']
        self.ratio_capture = args['ratio_capture']
        self.train_folder = args['train_folder']
        self.valid_folder = args['valid_folder']
        self.test_folder = args['test_folder']
        self.set_folder = args['set_folder']
        self.cascade_path = args['cascade_path']

    @staticmethod
    def str_join(paths: []):
        """
        Concatenate a list of paths
        Attributes:
            paths: List of paths.
        """
        return "/".join(paths)

    @staticmethod
    def create_folder(path: str):
        """
        Create folder in the path
        Attributes:
            paths: Path where will create the folder.
        """
        try:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except:
            print("An error occured.")

    @staticmethod
    def delete_folder(path: str):
        """
        Create folder in the path
        Attributes:
            paths: Path where will create the folder.
        """
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except:
            print("An error occured.")

    @staticmethod
    def get_name_folder(path: str, opts: list = 1):
        """
        Get a list with the elements of a folder
        Attributes:
            path: Folder with the elements
            opts : Mode of search
            1 : Return only folders
            otherwize : Return elements in path folder
        """
        try:
            folder_list = next(os.walk(path))
            folder_list = folder_list[1] if opts == 1 else folder_list[2]
            return folder_list
        except TypeError:
            print(TypeError)

    def video_to_frames(self, path_video: str, label: str, verbose: bool=False):
        """
        Convert video in frames and save it in a specific label folder
        Attributes:
            path_video: path where is the video to process
            label: The label(class) of the video
        """

        video = cv2.VideoCapture(path_video)

        success,  image = video.read()
        label_folder = Preprocessing.str_join([self.data_folder, label])
        timer = 0
        Preprocessing.create_folder(label_folder)
        while success:
            video.set(cv2.CAP_PROP_POS_MSEC, timer * self.ratio_capture)
            image_file_name = Preprocessing.str_join(
                [label_folder, str(datetime.now()) + '.jpg'])
            if verbose:
                print("Image file name", image_file_name)
            cv2.imwrite(image_file_name, image)
            success, image = video.read()
            timer += 1
        print("All  videos was convert in frames")

    @staticmethod
    def faces_from_image(args: list, mode: str="hog", cascade_path: str=""):
        """
        This method segment the faces in a image
        Attributes:
            args :
                -> args["files"] : list of the image paths
                -> args["label_data_path"] : Path of the label(class)
                -> args["label_face_path"] : Where we will save the images of the faces
            mode:
                -> haar : Use Haar to segment the face
                -> hog : Use Hog to segment the face
        """
        files = args["files"]
        label_data_path = args["label_data_path"]
        label_face_path = args["label_face_path"]
        for image_path in files:
            path_image = Preprocessing.str_join([label_data_path, image_path])
            image = cv2.imread(path_image)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            args = {
                "image": image_gray,
                "path_image": path_image,
                "label_path": label_face_path,
                "cascade_path": cascade_path
            }
            try:
                {
                    "haar": lambda args: Preprocessing.haar(args),
                    "hog": lambda args: Preprocessing.hog(args)
                }[mode](args)

            except:
                print("An error occured.")

    @staticmethod
    def hog(args: {}):
        """
        Hog method
        Attributes : 
            path_image:   Path image that will be processing.
            label_face_path: Folder where the face will save
        """
        imagen = args["image"]
        path_image = args["path_image"]
        label_face_path = args["label_path"]

        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(imagen, 1)

        for i, face_rect in enumerate(detected_faces):
            ima = Image.open(path_image)
            x_size = face_rect.left()
            y_size = face_rect.top()
            w_size = face_rect.right()
            h_size = face_rect.bottom()

            box = (x_size, y_size, w_size, h_size)
            face = ima.crop(box).resize((299, 299))
            face_file_path = Preprocessing.str_join(
                [label_face_path, str(datetime.now()) + str(i) + ".jpg"])
            face.save(face_file_path)
        return True

    @staticmethod
    def haar(args: {}):
        imagen = args["image"]
        path_image = args["path_image"]
        label_face_path = args["label_path"]
        cascade_path_xml = args["cascade_path"]

        face_detector = cv2.CascadeClassifier(cascade_path_xml)
        detected_faces = face_detector.detectMultiScale(imagen, 1.3, 5)
        for (x, y, w, h) in detected_faces:
            ima = Image.open(path_image)
            center_x = x + w/2
            center_y = y + h/2
            b_dim = min(max(w, h)*1.2, ima.width, ima.height)
            box = ((center_x-b_dim/2), (center_y-b_dim/2),
                   (center_x+b_dim/2), (center_y+b_dim/2))
            face = ima.crop(box).resize((224, 224))
            face_file_path = Preprocessing.str_join(
                [label_face_path, str(datetime.now()) + ".jpg"])
            face.save(face_file_path)
        return True

    def process_videos(self, verbose: bool=False):
        """
        Process each video and convert each frame in an image
        """
        videos = Preprocessing.get_name_folder(self.video_folder)

        for label in videos:
            video_label_path = Preprocessing.str_join(
                [self.video_folder, label])
            video_labels = Preprocessing.get_name_folder(video_label_path, 2)
            if verbose:
                print(videos)
                print('Video Label Path : ', video_label_path, video_labels)

            for video_label in video_labels:
                if verbose:
                    print(video_label)
                self.video_to_frames(
                    Preprocessing.str_join([video_label_path, video_label]), label, verbose)

    def process_images(self, mode: str="hog"):
        """
        fadfaD
        """
        labels = list(Preprocessing.get_name_folder(self.data_folder, 1))

        for label in labels:
            label_data_path = Preprocessing.str_join([self.data_folder, label])
            label_face_path = Preprocessing.str_join(
                [self.faces_folder, label])

            Preprocessing.create_folder(label_face_path)
            label_images = Preprocessing.get_name_folder(label_data_path, 2)

            args = {
                "files": label_images,
                "label_data_path": label_data_path,
                "label_face_path": label_face_path
            }
            Preprocessing.faces_from_image(args, mode, self.cascade_path)

    def delete_folders_dataset(self):
        Preprocessing.delete_folder(self.data_folder)
        Preprocessing.delete_folder(self.faces_folder)
        Preprocessing.delete_folder(self.set_folder)

    def create_folders_dataset(self):
        Preprocessing.create_folder(self.data_folder)
        Preprocessing.create_folder(self.faces_folder)
        Preprocessing.create_folder(self.set_folder)
        Preprocessing.create_folder(self.train_folder)
        Preprocessing.create_folder(self.test_folder)
        Preprocessing.create_folder(self.valid_folder)

    def create_sets(self, augmentation):
        """
        SAFDA
        """
        Preprocessing.create_folder(self.set_folder)
        labels = Preprocessing.get_name_folder(self.faces_folder, 1)
        for label in labels:
            train_label_path = Preprocessing.str_join(
                [self.train_folder, label])
            valid_label_path = Preprocessing.str_join(
                [self.valid_folder, label])
            test_label_path = Preprocessing.str_join(
                [self.test_folder, label])

            Preprocessing.create_folder(train_label_path)
            Preprocessing.create_folder(valid_label_path)
            Preprocessing.create_folder(test_label_path)

            label_path = Preprocessing.str_join([self.faces_folder, label])
            faces_label = Preprocessing.get_name_folder(label_path, 2)
            array_images_label = np.array(faces_label)
            np.random.shuffle(array_images_label)

            array_images_label, img_test = array_images_label[:(math.floor(len(
                array_images_label)*0.3))], array_images_label[(math.floor(len(array_images_label)*0.3)):]

            img_train, img_val = train_test_split(
                array_images_label, test_size=0.3)

            for img_t in img_train:
                shutil.copy2(Preprocessing.str_join(
                    [label_path, img_t]), train_label_path)
                if augmentation:
                    im = Image.open(
                        Preprocessing.str_join([label_path, img_t]))
                    img_aug = Preprocessing.augmentation(np.asarray(im))
                    for img in range(len(img_aug)):
                        Image.fromarray(img_aug[img]).save(Preprocessing.str_join(
                            [train_label_path, "aug-img-"+str(img) + str(datetime.now())+".jpg"]))

            for img_v in img_val:
                shutil.copy2(Preprocessing.str_join(
                    [label_path, img_v]), valid_label_path)

            for img_tst in img_test:
                shutil.copy2(Preprocessing.str_join(
                    [label_path, img_tst]), test_label_path)

    def generate_dataset(self, overwrite: bool =True, Augmentation: bool = False, mode: str='hog'):
        """sfsadfsd"""
        if overwrite:
            self.delete_folders_dataset()
        self.create_folders_dataset()
        self.process_videos()
        self.process_images(mode)
        self.create_sets(Augmentation)

    @staticmethod
    def augmentation(image):

        augmentadores = [
            iaa.Add(-10),
            iaa.Add(45),
            iaa.Add(80),
            iaa.GaussianBlur(0.50),
            iaa.GaussianBlur(1.0),
            iaa.Dropout(0.03),
            iaa.Dropout(0.05),
            iaa.Dropout(0.10),
            iaa.ContrastNormalization(0.5),
            iaa.ContrastNormalization(1.2),
            iaa.PerspectiveTransform(0.075),
            iaa.PerspectiveTransform(0.100),
            iaa.PerspectiveTransform(0.125),
            iaa.Grayscale(alpha=1.0),
            iaa.Grayscale(alpha=0.5),
            iaa.Grayscale(alpha=0.2),
            iaa.CoarsePepper(size_percent=0.30),
            iaa.CoarsePepper(size_percent=0.02),
            iaa.CoarsePepper(size_percent=0.1),
            iaa.SaltAndPepper(p=0.05),
            iaa.SaltAndPepper(p=0.03),
            iaa.Affine(scale=0.5),
        ]
        ImageArray = list()
        for i in range(len(augmentadores)):
            seq = iaa.Sequential(augmentadores[i])
            img_aug = seq.augment_image((np.asarray(image)))
            ImageArray.append(img_aug)
        return ImageArray
