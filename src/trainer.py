from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import normalize, to_categorical
from sklearn.metrics import classification_report, confusion_matrix
# import hypertune

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
# import shutil
from datetime import datetime

print("Tensorflow is running on following devices : ")
print(device_lib.list_local_devices())

# IMG_WIDTH = 24
# IMG_HEIGHT = 24
# InceptionV3_IMG_WIDTH = 229
# InceptionV3_IMG_HEIGHT = 229

def view_images(x_train):
    #view few images 
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i])
    plt.show()

def load_single_raw_image(image_path, image_size):
    X_dataset = []
    # for i in range(df.shape[0]):
    img = image.load_img(image_path, target_size=(image_size , image_size, 3))
    img = image.img_to_array(img)
    img = img/255.
    img = np.expand_dims(img, axis=0)
    return img
    # X_dataset.append(img)    
    # X = np.array(X_dataset)

def build_my_model3(nbr_classes, input_image_size):
    print(f"[INFO] build_my_model3 nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(input_image_size, input_image_size, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def build_my_model2(nbr_classes, input_image_size):
    print(f"[INFO] build_my_model2 nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', input_shape=(input_image_size, input_image_size, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
    # model.add(Dropout(0.4)) # dropout 避免Overfitting
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def build_my_model1_without_kernelInit(nbr_classes, input_image_size):
    print(f"[INFO] build_my_model1_without_kernelInit nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(input_image_size, input_image_size, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def build_my_model1(nbr_classes, input_image_size):
    print(f"[INFO] build_my_model1 nbr_classes: {nbr_classes}")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(input_image_size, input_image_size, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def build_model_based_on_inceptionv3(nbr_classes, input_image_size):
    print(f"[INFO] build_model_based_on_inceptionv3 nbr_classes: {nbr_classes}")
    base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(input_image_size, input_image_size, 3)))

    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(nbr_classes, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def build_data_pipelines(batch_size, train_data_path, val_data_path, eval_data_path, input_image_size):

    train_augmentor = ImageDataGenerator(
        rescale = 1. / 255,
        # rotation_range=25,
        # zoom_range=0.15,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.15,
        # horizontal_flip=True,
        # fill_mode="nearest"
    )

    val_augmentor = ImageDataGenerator(
        rescale = 1. / 255
    )

    train_generator = train_augmentor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(input_image_size, input_image_size),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
        # save_to_dir=os.path.join(train_data_path,'augmented')
    )

    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(input_image_size, input_image_size),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    eval_generator = val_augmentor.flow_from_directory(
        eval_data_path,
        class_mode="categorical",
        target_size=(input_image_size, input_image_size),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, eval_generator

def get_number_of_imgs_inside_folder(directory):

    totalcount = 0

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in [".png", ".jpg", ".JPG", ".jpeg"]:
                totalcount = totalcount + 1
    return totalcount

def plot_training_curve(history):
    #plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def train(path_to_data, batch_size, epochs, learning_rate, input_image_size, build_model_func, isEarlyStopping=False, show_graph=False):
    print(f"[INFO] batch_size: {batch_size}")
    print(f"[INFO] epochs: {epochs}")
    print(f"[INFO] learning_rate: {learning_rate}")
    print(f"[INFO] input_image_size: {input_image_size}")
    print(f"[INFO] Use build_model_func: {build_model_func.__name__}")
    print(f"[INFO] isEarlyStopping: {isEarlyStopping}")

    path_train_data = os.path.join(path_to_data, 'train')
    path_val_data = os.path.join(path_to_data, 'validation')
    path_eval_data = os.path.join(path_to_data, 'test')

    total_train_imgs = get_number_of_imgs_inside_folder(path_train_data)
    total_val_imgs = get_number_of_imgs_inside_folder(path_val_data)
    total_eval_imgs = get_number_of_imgs_inside_folder(path_eval_data)

    print(f"[INFO] total_train_imgs: {total_train_imgs}")
    print(f"[INFO] total_val_imgs: {total_val_imgs}")
    print(f"[INFO] total_eval_imgs: {total_eval_imgs}")
    
    train_generator, val_generator, eval_generator = build_data_pipelines(
        batch_size=batch_size,
        train_data_path=path_train_data,
        val_data_path=path_val_data,
        eval_data_path=path_eval_data,
        input_image_size=input_image_size
    )

    classes_dict = train_generator.class_indices
    nbr_classes = len(classes_dict.keys())
    classes_dict.keys()
    print(f"[INFO] classes_dict.keys(): {classes_dict.keys()}")
    print(f"[INFO] Total number of classes: {nbr_classes}")

    model = build_model_func(nbr_classes, input_image_size)
    
    optimizer = Adam(lr=learning_rate) # le-5
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    callbacks = []

    # # Use Mode = max for accuracy and min for loss. 
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    # # This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    if isEarlyStopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        callbacks.append(early_stopping)

    path_to_save_model = './output/saved_models'
    if not os.path.isdir(path_to_save_model):
        os.makedirs(path_to_save_model)

    # ModelCheckpoint callback saves a model at some interval. 
    fileName = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"   # File name includes epoch and validation accuracy.
    path_to_save_model = os.path.join(path_to_save_model, fileName)
    print(f"[DEBUG] path_to_save_model: {path_to_save_model}")

    ckpt_saver = ModelCheckpoint(
        path_to_save_model,
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )
    callbacks.append(ckpt_saver)

    print("[INFO] Start Training...")
    history = model.fit(                    # model.fit_generator is deprecated
        train_generator,
        steps_per_epoch=total_train_imgs // batch_size,
        validation_data=val_generator,
        validation_steps=total_val_imgs // batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    print("[INFO] Training Finish!")

    model.save('./output/trained_model.h5')  # always save your weights after training or during training
    print("[INFO] Save trained model successfully!")

    print("[INFO] Evaluation phase...")
    predictions = model.predict(eval_generator)         # model.predict_generator is deprecated
    predictions_idxs = np.argmax(predictions, axis=1)

    print(f"[INFO] eval_generator.classes: {eval_generator.classes}")

    my_classification_report = classification_report(eval_generator.classes, predictions_idxs, 
                                                        target_names=eval_generator.class_indices.keys())

    my_confusion_matrix = confusion_matrix(eval_generator.classes, predictions_idxs)

    print("[INFO] Classification report : ")
    print(my_classification_report)
    print()

    print("[INFO] Confusion matrix : ")
    print(my_confusion_matrix)
    print()

    # evaluating the model
    print("[INFO] Starting evaluation using model.evaluate_generator...")
    # train_loss, train_acc = model.evaluate_generator(train_generator, steps=16)
    # validation_loss, val_acc = model.evaluate_generator(val_generator, steps=16)
    train_loss, train_acc = model.evaluate(train_generator)
    val_loss, val_acc = model.evaluate(val_generator)
    print('[INFO] Train Accuracy: %.3f, Validation Accuracy: %.3f' % (train_acc, val_acc))
    print('[INFO] Train Loss: %.3f, Validation Loss: %.3f' % (train_loss, val_loss))

    # plot training history
    if show_graph:
        print("Values stored in history are ... \n", history.history)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.show()
        print("Done evaluating!")
    return history, model

    # print("Starting evaluation using model.evaluate_generator")
    # scores = model.evaluate_generator(eval_generator)
    # print("Done evaluating!")
    # loss = scores[0]
    # print(f"loss for hyptertune = {loss}")
    
    # now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # zipped_folder_name = f'trained_model_{now}_loss_{loss}'
    
    # hpt = hypertune.HyperTune()
    # hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='loss', 
    #                                         metric_value=loss, global_step=epochs)


def retrain():
    pass
    """
    #To continue training, by modifying weights to existing model.
    #The saved model can be reinstated.
    from keras.models import load_model
    new_model = load_model('malaria_augmented_model.h5')
    results = new_model.evaluate_generator(validation_generator, steps=16)
    print(" validation loss and accuracy are", results)
    new_model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,    #The 2 slashes division return rounded integer
            epochs=5,
            validation_data=validation_generator,
            validation_steps=800 // batch_size,
            callbacks=callbacks_list)
    model.save('malaria_augmented_model_updated.h5') 
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size used by the deep learning model", 
                        default=2)
    parser.add_argument("--epochs", type=int, help="Epochs used by the deep learning model", 
                        default=20)
    parser.add_argument("--learning_rate", type=float, help="Batch size used by the deep learning model", 
                        default=1e-5)
    parser.add_argument("--build_model_func", type=str, help="Pass which model architecture you want to use")
    parser.add_argument("--input_image_size", type=str, help="The size of input image")
    args = parser.parse_args()

    path_to_data = "./dataset"

    train(path_to_data, args.batch_size, args.epochs, args.learning_rate, args.input_image_size, args.build_model_func)
    print("Training Finished!")