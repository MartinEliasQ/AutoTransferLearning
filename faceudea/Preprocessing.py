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
        except TypeError:
            print(TypeError)

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
        except TypeError:
            print(TypeError)

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

    def faces_from_image(self, args: list, mode: str):
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
            try:
                {
                    #   "haar": lambda img, path, label_path: self.haar(img, path, label_path),
                    "hog": lambda img, path, label_path: Preprocessing.hog(img, path, label_path)
                }[mode](image_gray, path_image, label_face_path)
            except print(0):
                pass

    @staticmethod
    def hog(imagen, path_image: str, label_face_path: str):
        """
        Hog method
        Attributes : 
            path_image:   Path image that will be processing.
            label_face_path: Folder where the face will save
        """
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

            for videoL in video_labels:
                if verbose:
                    print(videoL)
                self.video_to_frames(
                    Preprocessing.str_join([video_label_path, videoL]), label, verbose)

    def process_images(self, mode: str="hog"):
        """

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
            self.faces_from_image(args, mode)

    def create_sets(self):
        """

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

            for img_v in img_val:
                shutil.copy2(Preprocessing.str_join(
                    [label_path, img_v]), valid_label_path)

            for img_tst in img_test:
                shutil.copy2(Preprocessing.str_join(
                    [label_path, img_tst]), test_label_path)

    def Augmentation(self):
        pass

    def generate_dataset(self, overwrite: bool =True, Augmentation: bool = False):
        if overwrite:
            Preprocessing.delete_folder(self.data_folder)
            Preprocessing.delete_folder(self.faces_folder)
            Preprocessing.delete_folder(self.set_folder)
        Preprocessing.create_folder(self.data_folder)
        Preprocessing.create_folder(self.faces_folder)
        Preprocessing.create_folder(self.set_folder)
        Preprocessing.create_folder(self.train_folder)
        Preprocessing.create_folder(self.test_folder)
        Preprocessing.create_folder(self.valid_folder)
        self.process_videos()
        self.process_images()
        self.create_sets()
