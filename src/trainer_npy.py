import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import kerastuner as kt
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import random


def infer_dirs_as_labels(dirPath):
  labels = [name for name in os.listdir(dirPath) if os.path.isdir(os.path.join(dirPath,name))]
  dctLabel = {}
  for idx,name in enumerate(labels):
    dctLabel[name] = idx

  return dctLabel, len(labels)

# 將labels轉成onehot
def get_onehot_label(labels, num_classes):
  onehot_labels = keras.utils.to_categorical(labels, num_classes)
  return onehot_labels

def load_data(dirPath, dctLabel):
  lstClasses = os.listdir(dirPath)
  X_dataset = []
  Y_labels = []
  for className in lstClasses:
    classPath = os.path.join(dirPath, className)
    for fileName in os.listdir(classPath):
      filePath = os.path.join(classPath, fileName)
      _, ext = os.path.splitext(fileName)
      if ext in [".npy"]:
        data = np.load(filePath)
        data = data.reshape(data.shape[0], data.shape[1], 1)
        label = dctLabel[className]
        X_dataset.append(data)
        Y_labels.append(label)
  Y_onehot_labels = get_onehot_label(Y_labels, len(lstClasses))
  X_dataset = np.array(X_dataset)
  return X_dataset, Y_labels, Y_onehot_labels

def make_dataset(X_dataset, Y_onehot_labels, shuffle_buffer):
  ds = tf.data.Dataset.from_tensor_slices((X_dataset, Y_onehot_labels))

  # 合併圖片與label資料集
  # ds = tf.data.Dataset.zip((X_dataset, Y_onehot_labels))
  
  # 打散
  ds = ds.shuffle(shuffle_buffer,reshuffle_each_iteration=False)
  return ds

def train(path_to_data, batch_size, epochs, learning_rate, input_image_size, build_model_func, isEarlyStopping=False, show_graph=False):
    print(f"[INFO] batch_size: {batch_size}")
    print(f"[INFO] epochs: {epochs}")
    print(f"[INFO] learning_rate: {learning_rate}")
    print(f"[INFO] input_image_size: {input_image_size}")
    print(f"[INFO] Use build_model_func: {build_model_func.__name__}")
    print(f"[INFO] isEarlyStopping: {isEarlyStopping}")

    train_dir = os.path.join(path_to_data, 'train')
    validation_dir = os.path.join(path_to_data, 'validation')
    test_dir = os.path.join(path_to_data, 'test')

    dctLabel, num_classes = infer_dirs_as_labels(train_dir)
    X_train_dataset, Y_train_labels, Y_train_onehot_labels = load_data(train_dir, dctLabel)
    X_validation_dataset, Y_validation_labels, Y_validation_onehot_labels = load_data(validation_dir, dctLabel)
    X_test_dataset, Y_test_labels, Y_test_onehot_labels = load_data(test_dir, dctLabel)
    print(f"[INFO] dctLabel.keys(): {dctLabel.keys()}")
    print(f"[INFO] Total number of classes: {num_classes}")

    print(f"[INFO] total_train_imgs: {len(X_train_dataset)}")
    print(f"[INFO] total_val_imgs: {len(X_validation_dataset)}")
    print(f"[INFO] total_eval_imgs: {len(X_test_dataset)}")

    trainBufferSize = len(X_train_dataset)
    valBufferSize = len(X_validation_dataset)
    testBufferSize = len(X_test_dataset)
    train_ds = make_dataset(X_train_dataset, Y_train_onehot_labels, trainBufferSize)
    validation_ds = make_dataset(X_validation_dataset, Y_validation_onehot_labels, valBufferSize)
    test_ds = make_dataset(X_test_dataset, Y_test_onehot_labels, testBufferSize)

    train_ds = train_ds.batch(batch_size)
    validation_ds = validation_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    model = build_model_func(num_classes, input_image_size)
    
    print("[INFO] Model Summary")
    print("*"*30)
    print(model.summary())
    print("*"*30)

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
    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=callbacks)
    print("[INFO] Training Finish!")

    model.save('./output/trained_model.h5')  # always save your weights after training or during training
    print("[INFO] Save trained model successfully!")

    print("[INFO] Evaluation phase...")
    predictions = model.predict(test_ds)         # model.predict_generator is deprecated
    predictions_idxs = np.argmax(predictions, axis=1)

    print(f"[INFO] test_ds.classes: {test_ds.classes}")

    # print(classification_report(y_test, y_predict, target_names=class_names))
    # print(confusion_matrix(y_test, y_predict))

    my_classification_report = classification_report(Y_test_onehot_labels, predictions_idxs, 
                                                        target_names=test_ds.class_indices.keys())

    my_confusion_matrix = confusion_matrix(Y_test_onehot_labels, predictions_idxs)

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
    train_loss, train_acc = model.evaluate(train_ds)
    val_loss, val_acc = model.evaluate(validation_ds)
    test_loss, test_acc = model.evaluate(test_ds)
    print('[INFO] Train Accuracy: %.3f, Validation Accuracy: %.3f' % (train_acc, val_acc))
    print('[INFO] Train Loss: %.3f, Validation Loss: %.3f' % (train_loss, val_loss))
    print('[INFO] Test Loss: %.3f, Test Accuracy: %.3f' % (test_loss, test_acc))

    # # plot training history
    # if show_graph:
    #     print("Values stored in history are ... \n", history.history)
    #     plt.plot(history.history['loss'], label='train')
    #     plt.plot(history.history['val_loss'], label='val')
    #     plt.legend()
    #     plt.show()
    #     print("Done evaluating!")
    
    return history, model

def train_with_easy(path_to_data, batch_size, epochs, learning_rate, input_image_size, build_model_func, isEarlyStopping=False, show_graph=False):
    print(f"[INFO] batch_size: {batch_size}")
    print(f"[INFO] epochs: {epochs}")
    print(f"[INFO] learning_rate: {learning_rate}")
    print(f"[INFO] input_image_size: {input_image_size}")
    print(f"[INFO] Use build_model_func: {build_model_func.__name__}")
    print(f"[INFO] isEarlyStopping: {isEarlyStopping}")

    train_dir = os.path.join(path_to_data, 'train')
    validation_dir = os.path.join(path_to_data, 'validation')
    test_dir = os.path.join(path_to_data, 'test')

    dctLabel, num_classes = infer_dirs_as_labels(train_dir)
    X_train_dataset, Y_train_labels, Y_train_onehot_labels = load_data(train_dir, dctLabel)
    X_validation_dataset, Y_validation_labels, Y_validation_onehot_labels = load_data(validation_dir, dctLabel)
    X_test_dataset, Y_test_labels, Y_test_onehot_labels = load_data(test_dir, dctLabel)
    print(f"[INFO] dctLabel.keys(): {dctLabel.keys()}")
    print(f"[INFO] Total number of classes: {num_classes}")

    print(f"[INFO] total_train_imgs: {len(X_train_dataset)}")
    print(f"[INFO] total_val_imgs: {len(X_validation_dataset)}")
    print(f"[INFO] total_eval_imgs: {len(X_test_dataset)}")

    trainBufferSize = len(X_train_dataset)
    valBufferSize = len(X_validation_dataset)
    testBufferSize = len(X_test_dataset)
    train_ds = make_dataset(X_train_dataset, Y_train_onehot_labels, trainBufferSize)
    validation_ds = make_dataset(X_validation_dataset, Y_validation_onehot_labels, valBufferSize)
    test_ds = make_dataset(X_test_dataset, Y_test_onehot_labels, testBufferSize)

    train_ds = train_ds.batch(batch_size)
    validation_ds = validation_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    model = build_model_func(num_classes, input_image_size)
    
    print("[INFO] Model Summary")
    print("*"*30)
    print(model.summary())
    print("*"*30)

    optimizer = Adam(learning_rate=learning_rate) # le-5
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("[INFO] Start Training...")
    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
    print("[INFO] Training Finish!")
    
    return history, model

def train_hp(path_to_data, epochs, input_image_size, build_model_class, EARLY_STOPPING_PATIENCE, OUTPUT_PATH, tuner_method, max_trials=5, overwrite=False):
  print(f"[INFO] epochs: {epochs}")
  print(f"[INFO] input_image_size: {input_image_size}")
  print(f"[INFO] Use build_model_class: {build_model_class.__name__}")

  train_dir = os.path.join(path_to_data, 'train')
  validation_dir = os.path.join(path_to_data, 'validation')
  test_dir = os.path.join(path_to_data, 'test')

  dctLabel, num_classes = infer_dirs_as_labels(train_dir)
  X_train_dataset, Y_train_labels, Y_train_onehot_labels = load_data(train_dir, dctLabel)
  X_validation_dataset, Y_validation_labels, Y_validation_onehot_labels = load_data(validation_dir, dctLabel)
  X_test_dataset, Y_test_labels, Y_test_onehot_labels = load_data(test_dir, dctLabel)
  print(f"[INFO] dctLabel.keys(): {dctLabel.keys()}")
  print(f"[INFO] Total number of classes: {num_classes}")

  print(f"[INFO] total_train_imgs: {len(X_train_dataset)}")
  print(f"[INFO] total_val_imgs: {len(X_validation_dataset)}")
  print(f"[INFO] total_eval_imgs: {len(X_test_dataset)}")

  trainBufferSize = len(X_train_dataset)
  valBufferSize = len(X_validation_dataset)
  testBufferSize = len(X_test_dataset)
  train_ds = make_dataset(X_train_dataset, Y_train_onehot_labels, trainBufferSize)
  validation_ds = make_dataset(X_validation_dataset, Y_validation_onehot_labels, valBufferSize)
  test_ds = make_dataset(X_test_dataset, Y_test_onehot_labels, testBufferSize)

  es = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

  build_model_obj = build_model_class(num_classes, input_image_size)

  if tuner_method == "hyperband":
    # instantiate the hyperband tuner object
    print("[INFO] instantiating a hyperband tuner object...")
    tuner = kt.Hyperband(
      build_model_obj,
      objective="val_accuracy",
      max_epochs=epochs,
      factor=3,
      seed=42,
      directory=OUTPUT_PATH,
      project_name=tuner_method,
      overwrite=overwrite)
  elif tuner_method == "random":
    # instantiate the random search tuner object
    print("[INFO] instantiating a random search tuner object...")
    tuner = kt.RandomSearch(
      build_model_obj,
      objective="val_accuracy",
      max_trials=5,
      executions_per_trial=1,
      # seed=42,
      directory=OUTPUT_PATH,
      project_name=tuner_method,
      overwrite=overwrite)
  else:
    # instantiate the bayesian optimization tuner object
    print("[INFO] instantiating a bayesian optimization tuner object...")
    tuner = kt.BayesianOptimization(
      build_model_obj,
      objective="val_accuracy",
      max_trials=10,
      seed=42,
      directory=OUTPUT_PATH,
      project_name=tuner_method,
      overwrite=overwrite)

  print("[INFO] tuner.search_space_summary()...")
  print(tuner.search_space_summary())

  # perform the hyperparameter search
  print("[INFO] performing hyperparameter search...")
  tuner.search(
    x=X_train_dataset, y=Y_train_onehot_labels,
    validation_data=(X_validation_dataset, Y_validation_onehot_labels),
    # train_ds,
    # validation_data=validation_ds
    # batch_size=batch_size,
    epochs=epochs,
    callbacks=[es]
  )

  # grab the best hyperparameters
  print("[INFO] tuner.get_best_hyperparameters()[0].values...")
  bestHP = tuner.get_best_hyperparameters()[0]
  print(bestHP.values)
  # print("[INFO] optimal number of filters in conv_1 layer: {}".format(bestHP.get("conv_1")))
  # print("[INFO] optimal number of filters in conv_2 layer: {}".format(bestHP.get("conv_2")))
  # print("[INFO] optimal number of units in dense layer: {}".format(bestHP.get("dense_units")))
  # print("[INFO] optimal learning rate: {:.4f}".format(bestHP.get("learning_rate")))

  print("[INFO] tuner.results_summary()...")
  print(tuner.results_summary())

  # get the best model
  model=tuner.get_best_models()[0]

  # print final model structure
  print("[INFO] Final model structure")
  print(model.summary())

  # return the model
  return model
