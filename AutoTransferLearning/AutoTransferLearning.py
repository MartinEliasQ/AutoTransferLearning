from functools import reduce

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import models, layers, optimizers


class Faceudea(object):
    """asdfasfd"""

    def __init__(self, args: {}):
        self.pretrained_model = args["pretrained_model"]
        self.image_size = args["image_size"]
        self.freeze_layers = args["freeze_layers"]

        self.preprocessing = args["preprocessing"]

    def freeze_layers_model(self):
        """dfasdf"""
        for layer in self.pre_model.layers[:-self.freeze_layers]:
            layer.trainable = False
        return True

    def select_model(self):
        """dfsad"""
        print(self.pretrained_model)
        model = self.pretrained_model
        if model == "VGG16":
            return VGG16(weights="imagenet", include_top=False,
                         input_shape=(self.image_size, self.image_size, 3))
        if model == "VGG19":
            return VGG19(weights="imagenet", include_top=False,
                         input_shape=(self.image_size, self.image_size, 3))
        if model == "InceptionV3":
            return InceptionV3(weights="imagenet", include_top=False,
                               input_shape=(self.image_size, self.image_size, 3))
        if model == "ResNet50":
            return ResNet50(weights="imagenet", include_top=False,
                            input_shape=(self.image_size, self.image_size, 3))

    def pretrain_model(self):
        """dafasdf"""
        try:
            self.pre_model = self.select_model()
            self.freeze_layers_model()
            print("Layers Freezed OK!")
        except:
            print("An error occured in the pretrain_model.")

    def add_dense(self, config_layer):
        """asfasdf"""
        number_neurons = config_layer[1]
        func_activation = config_layer[2]
        self.classifier.add(layers.Dense(
            number_neurons, activation=func_activation))
        print("Added Dense Layer")

    def add_dropout(self, config_layer):
        """dfasd"""
        rate = config_layer[1]
        self.classifier.add(layers.Dropout(float(rate)))
        print("Added Dropout Layer")

    def add_layers(self, layer_list):

        print("This is the layer:", layer_list)
        """dfsd"""
        {
            "Dense": lambda layer: self.add_dense(layer),
            "Dropout": lambda layer: self.add_dropout(layer)
        }[layer_list[0]](layer_list)
        print("Layers Added")

    def create_classifier(self, layer_list: list=list()):
        """asfsdf"""
        self.classifier = models.Sequential()
        try:
            self.pretrain_model()
        except:
            print("An error occured in the pretrained.")
        self.classifier.add(self.pre_model)
        self.classifier.add(layers.Flatten())

        for layer in layer_list:
            self.add_layers(layer)
        self.classifier.compile(loss="categorical_crossentropy",
                                optimizer=optimizers.RMSprop(1e-4), metrics=["acc"])

    def train(self, batch_train, batch_valid, batch_test, epochs):
        self.train_datagen = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.4, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            self.preprocessing.train_folder,
            target_size=(self.image_size, self.image_size),
            class_mode="categorical",
            shuffle=True,
            batch_size=batch_train)

        self.validation_generator = self.test_datagen.flow_from_directory(
            self.preprocessing.valid_folder,
            batch_size=batch_valid,
            class_mode="categorical",
            target_size=(self.image_size, self.image_size))

        self.test_generator = self.test_datagen.flow_from_directory(
            self.preprocessing.test_folder,
            batch_size=batch_valid,
            class_mode="categorical",
            target_size=(self.image_size, self.image_size))

        self.model = self.classifier.fit_generator(self.train_generator,
                                                   steps_per_epoch=self.train_generator.samples/batch_train,
                                                   epochs=epochs,
                                                   validation_data=self.validation_generator,
                                                   validation_steps=self.validation_generator.samples /
                                                   self.validation_generator.batch_size
                                                   )
        print("Done!")
