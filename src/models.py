from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt

'''
I once got the same error. this error comes when your input size is smaller than 
your number of down samples(max pooling layer).

In other words,for example, 
when you apply max pooling layer of,say, (2,2) number of times to an input size of, 
say (256,256,3), there comes a point when your input size becomes (1,1,...) 
(just an example to understand). 
And at this point when Maxpool of size(2,2) is applied the input size becomes negative.

There are two simple solutions:-

Increase your input size, or
Decrease your maxpooling layer
I personally prefer the 1st solution.
'''

class MyHyperModel(kt.HyperModel):

    def __init__(self, nbr_classes, input_size):
        self.nbr_classes = nbr_classes
        self.input_size = input_size

    def build(self, hp):
        # initialize the model along with the input shape and channel
        model = keras.Sequential()
        chanDim = -1

        model.add(keras.Input(shape=self.input_size))
        model.add(layers.ZeroPadding2D(3))
        model.add(Conv2D(
            hp.Int("conv_1_input_units", min_value=8, max_value=64, step=8),
            (3, 3), padding="same"))
        # model.add(Conv2D(
        #     hp.Int("conv_1_input_units", min_value=8, max_value=64, step=8),
        #     (3, 3), padding="same", input_shape=self.input_size))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        for i in range(hp.Int("first_conv_n_layers", 0, 3)):
            model.add(Conv2D(
                hp.Int(f"first_conv_{i}_units", min_value=8, max_value=64, step=8),
                (3, 3), padding="same"))
            model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        for i in range(hp.Int("sec_conv_n_layers", 1, 3)):
            model.add(Conv2D(
                hp.Int(f"sec_conv_{i}_units", min_value=8, max_value=64, step=8),
                (3, 3), padding="same"))
            model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(hp.Int("dense_1_units", min_value=256,
            max_value=1024, step=256)))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))

        for i in range(hp.Int("dense_n_layers", 0, 3)):
            model.add(Dense(hp.Int(f"dense_{i}_units", min_value=32,
                max_value=768, step=32)))
            model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(self.nbr_classes))
        model.add(Activation("softmax"))

        # initialize the learning rate choices and optimizer
        hp_lr = hp.Choice("learning_rate",
            values=[1e-1, 1e-2, 5e-3, 1e-3, 5e-4])
        
        hp_optimizer = hp.Choice('optimizer', values=['sgd', 'rmsprop', 'adam'])

        if hp_optimizer == 'sgd':
            print(f"[INFO] using optimizer {hp_optimizer}")
            optimizer = SGD(learning_rate=hp_lr)
        elif hp_optimizer == 'rmsprop':
            print(f"[INFO] using optimizer {hp_optimizer}")
            optimizer = RMSprop(learning_rate=hp_lr)
        elif hp_optimizer == 'adam':
            print(f"[INFO] using optimizer {hp_optimizer}")
            optimizer = Adam(learning_rate=hp_lr)
        else:
            print(f"[ERROR] using optimizer {hp_optimizer}")
            raise
        # opt = Adam(learning_rate=hp_lr)
        
        # compile the model
        model.compile(optimizer=hp_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [4, 8, 16, 32]),
            **kwargs,
        )

def build_easy_model_var25(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var24(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var23(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var22(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var21(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var20(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var19(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var18(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var17(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var16(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var11 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.ZeroPadding2D(3),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            # layers.Dense(32, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var15(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var11 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.ZeroPadding2D(3),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            # layers.Dense(32, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var14(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var11 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.ZeroPadding2D(3),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            # layers.Dense(32, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var13(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var11 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.ZeroPadding2D(3),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            # layers.Dense(32, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var12(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var11 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.ZeroPadding2D(3),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            # layers.Dense(32, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var11(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var11 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.ZeroPadding2D(3),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            # layers.Dense(32, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var10(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var10 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(32, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var93(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var92(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var91(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var9(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var9 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var8(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var8 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var7(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var7 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            # layers.Dropout(0.1),
            layers.Dense(8, activation='relu'),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var6(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var6 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            # layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            # layers.Dense(32, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var5(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var5 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var4(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var4 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var3(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var3 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var2(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var2 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_var1(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_var1 nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model_without_dropout(nbr_classes, input_size):
    print(f"[INFO] build_easy_model_without_dropout nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model(nbr_classes, input_size):
    print(f"[INFO] build_easy_model nbr_classes: {nbr_classes}")
    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(nbr_classes, activation="softmax"),
        ]
    )
    return model

def build_easy_model3(nbr_classes, input_size):
    print(f"[INFO] build_easy_model3 nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_size))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
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

def build_easy_model2(nbr_classes, input_size):
    print(f"[INFO] build_easy_model2 nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=input_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    # model.add(Dropout(0.4)) # dropout 避免Overfitting
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def build_easy_model1(nbr_classes, input_size):
    print(f"[INFO] build_my_model1 nbr_classes: {nbr_classes}")

    model = Sequential()
    model.add(keras.Input(shape=input_size)),
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_size))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def build_my_model3(nbr_classes, input_size):
    print(f"[INFO] build_my_model3 nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_size))
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

def build_my_model2(nbr_classes, input_size):
    print(f"[INFO] build_my_model2 nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', input_shape=input_size, activation='relu'))
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

def build_my_model1_without_kernelInit(nbr_classes, input_size):
    print(f"[INFO] build_my_model1_without_kernelInit nbr_classes: {nbr_classes}")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_size))
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

def build_my_model1_without_padding(nbr_classes, input_size):
    print(f"[INFO] build_my_model1_without_padding nbr_classes: {nbr_classes}")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_size))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(nbr_classes, activation='softmax'))
    return model

def build_my_model1(nbr_classes, input_size):
    print(f"[INFO] build_my_model1 nbr_classes: {nbr_classes}")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_size))
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