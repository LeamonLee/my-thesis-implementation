def importF
    import tensorflow
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    import keras

    import numpy as np
    import matplotlib.pyplot as plt
    # import mpl_finance as mpf
    import pandas as pd
    import talib
    import seaborn as sns
    import sklearn
    from PIL import Image
    # import cv2
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import matplotlib.pyplot as plt
    from pyts.image import RecurrencePlot, MarkovTransitionField, \
        GramianAngularField
    import keras.utils as np_utils
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import Normalizer

    from keras.models import Sequential, Input, Model
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, \
        concatenate
    from keras.optimizers import Adam
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

    from keras.callbacks import EarlyStopping 

def load_data
    df = pd.read_csv('txf_new.csv')  # 讀入data, 並改為Talib可用的"欄位名稱"
    df.iloc[:, 2:7] = df.iloc[:, 2:7].astype('double')
    # df = df.rename(columns={'Index': 'DateTime'})
    df = df.rename(columns={' <Open>': 'Open'})
    df = df.rename(columns={' <High>': 'High'})
    df = df.rename(columns={' <Low>': 'Low'})
    df = df.rename(columns={' <Close>': 'Close'})
    df = df.rename(columns={' <Volume>': 'Volume'})
    df_arr = np.asarray(df.iloc[:, 2:])
    df_arr.shape  # 37021*5 df轉成np.array___包含"開、高、低、收、量" 的 array

    d_ma = talib.MA(np.array(df['Close']), timeperiod=25, matype=0)
    d_ma = np.reshape(d_ma, (1, -1))

    d_ma100 = talib.MA(np.array(df['Close']), timeperiod=100, matype=0)
    d_ma100 = np.reshape(d_ma100, (1, -1))

    d_sar = talib.SAREXT(df['High'], df['Low'])
    d_sar = np.array(d_sar)
    d_sar = np.reshape(d_sar, (1, -1))

    d_ultosc = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=5,
                            timeperiod2=15, timeperiod3=25)
    d_ultosc = np.array(d_ultosc)
    d_ultosc = np.reshape(d_ultosc, (1, -1))

    d_obv = talib.OBV(df['Close'], df['Volume'])
    d_obv = np.array(d_obv)
    d_obv = np.reshape(d_obv, (1, -1))

    d_atr = talib.ATR(np.array(df['High']), np.array(df['Low']),
                      np.array(df['Close']), timeperiod=25)
    d_atr = np.reshape(d_atr, (1, -1))

    sin = talib.HT_SINE(np.array(df['Close']))
    d_sin0 = sin[0]
    d_sin1 = sin[1]
    d_sin0 = np.reshape(d_sin0, (1, -1))
    d_sin1 = np.reshape(d_sin1, (1, -1))

    d_var = talib.VAR(np.array(df['Close']), timeperiod=25, nbdev=1)
    d_var = np.reshape(d_var, (1, -1))

    d_mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'],
                      timeperiod=25)
    d_mfi = np.array(d_mfi)
    d_mfi = np.reshape(d_mfi, (1, -1))

    d_ad = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    d_ad = np.array(d_ad)
    d_ad = np.reshape(d_ad, (1, -1))

    d_adx = talib.ADX(np.array(df['High']), np.array(df['Low']),
                      np.array(df['Close']), timeperiod=25)
    d_adx = np.reshape(d_adx, (1, -1))

    d_ma.shape, d_ma100.shape, d_sar.shape, d_ultosc.shape, d_obv.shape, d_atr.shape
    d_sin0.shape, d_sin1.shape, d_var.shape, d_mfi.shape, d_ad.shape, d_adx.shape
    ta_all = np.concatenate((d_ma, d_ma100, d_sar, d_ultosc, d_obv, d_atr,
                             d_sin0, d_sin1, d_var, d_mfi, d_ad, d_adx), axis=0)
    ta_all.shape
    ta_all = ta_all.T
    df_arr.shape, ta_all.shape  # 確定df_all跟ta_all的陣列是否一樣寬
    df_ta = np.concatenate((df_arr, ta_all), axis=1)  # df_ta就是all data+技術指標
    df_ta = df_ta[200:, :]  # 最前兩個月不要，因為指標會有nan
    # scaler = MinMaxScaler(feature_range=[0, 1])
    # df_tas = scaler.fit_transform(df_ta)   #經過正規化
    # df_tas=df_tas.T #陣列轉置90度
    # df_tas.shape
    df_ta = df_ta.T
    df_ta.shape
    # 17*36821 加了d_ma100
    np.save("df_x.npy", df_ta[:, 0:36620])

    y_pre = df_ta[3, :]
    y_pre.shape
    for yi in range(y_pre.shape[0] - 150): y_pre[yi] = y_pre[yi + 51] - y_pre[yi + 50]
    y_pre = y_pre[0:36620]  # 因為總共的列有36821，扣200後則變36620 （0-36619）
    # 若是y_pre >100 則為1 < -100 則為2 其餘為0
    y_pre.shape[0]
    for yj in range(y_pre.shape[0]):
        if y_pre[yj] > 0:
            y_pre[yj] = 1
        else:
            y_pre[yj] = 0

    y_pre = y_pre[0:36621]  # 因為總共的列有27463，扣200後則變27263 （0-27262)
    y_pre.shape
    sum(y_pre == 0)  # 16057
    sum(y_pre == 1)  # 50563
    np.save("y_pre.npy", y_pre)


def for_loop
    for ii in range(1,11):
        lr=0.00005*ii
        # locals()['cvscores_%s' % (ii)]
        locals()['cvscores_%s' % (ii)] = np.zeros((10, 12))
        for ai in range(0, 10):
            def cnn():
                model = Sequential()
                # CNN1
                model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
                                 input_shape=(24, 24, 5), activation='relu'))
                # MaxPooling1
                model.add(MaxPool2D(pool_size=(2, 2)))
                # CNN2
                model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
                                 activation='relu'))
                # MaxPooling2
                model.add(MaxPool2D(pool_size=(2, 2)))
                # CNN3
                model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
                                 activation='relu'))
                model.add(MaxPool2D(pool_size=(2, 2)))
                # dropout 避免Overfitting
                model.add(Dropout(0.4))
                # 平坦化
                model.add(Flatten())
                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.3))
                model.add(Dense(256, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(2, activation='sigmoid'))
                # 模型訓練
                adam = keras.optimizers.Adam(lr=lr, beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=None, decay=0.0,
                                             amsgrad=True)
                model.compile(loss='binary_crossentropy', optimizer=adam,
                              metrics=['accuracy'])
                return model

            xi = ai
            # total 的x 為 df_x.npy    y為 y_pre.ngy
            x_all = np.load("df_x.npy")
            y_all = np.load("y_pre.npy")
            x_all.shape, y_all.shape  # x=17,36620  y=36620,

            x = np.copy(x_all[3, :])  # 第3欄是收盤價
            x_ma25 = np.copy(x_all[5, :])
            x_ma100 = np.copy(x_all[6, :])
            x_atr = np.copy(x_all[10, :])
            x_adx = np.copy(x_all[16, :])

            y = y_all

            # 每2000一STEP 例如 train 0:10000 test 2000-12000

            xj = xi * 2000  # 間隔
            x1 = x[xj:xj + 20050]  # 100根，做圖i~~i+100
            x_ma25_1 = x_ma25[xj:xj + 20050]
            x_ma100_1 = x_ma100[xj:xj + 20050]
            x_atr_1 = x_atr[xj:xj + 20050]
            x_adx_1 = x_adx[xj:xj + 20050]

            y1 = y[xj:xj + 20050]

            x1.shape, y1.shape
            x1 = np.reshape(x1, (1, -1))
            normalizer = Normalizer().fit(x1)
            x2 = normalizer.transform(x1)

            x_ma25_1 = np.reshape(x_ma25_1, (1, -1))
            normalizer = Normalizer().fit(x_ma25_1)
            x_ma25_2 = normalizer.transform(x_ma25_1)

            x_ma100_1 = np.reshape(x_ma100_1, (1, -1))
            normalizer = Normalizer().fit(x_ma100_1)
            x_ma100_2 = normalizer.transform(x_ma100_1)

            x_atr_1 = np.reshape(x_atr_1, (1, -1))
            normalizer = Normalizer().fit(x_atr_1)
            x_atr_2 = normalizer.transform(x_atr_1)

            x_adx_1 = np.reshape(x_adx_1, (1, -1))
            normalizer = Normalizer().fit(x_adx_1)
            x_adx_2 = normalizer.transform(x_adx_1)

            x2 = np.ravel(x2.T)  # 再將X2轉置，並展開RAVEL
            x_ma25_2 = np.ravel(x_ma25_2.T)
            x_ma100_2 = np.ravel(x_ma100_2.T)
            x_atr_2 = np.ravel(x_atr_2.T)
            x_adx_2 = np.ravel(x_adx_2.T)

            x2.shape, x_ma25_2.shape, x_ma100_2.shape, x_atr_2.shape, x_adx_2.shape, y1.shape

            ox = "Close"  # 存npy 時的命名：若為"ma"則檔名為 gadf_ma.npy
            ic = 0  # i的 count
            np_a = np.zeros((20000, 576))
            x2.shape

            for i in range(0, x2.shape[0] - 50, 1):  # 每100根K畫一張圖
                x = x2[i:i + 50]
                x = x.reshape(1, -1)
                x = x * 255

                gadf = GramianAngularField(image_size=24, method='difference')
                x_gadf = gadf.fit_transform(x)

                x_gadf[0].shape  # 24*24=576
                # plt.imshow(x_gadf[0])
                a = np.reshape(x_gadf[0], (1, -1))
                # a.shape
                np_a[i] = a
                ic = ic + 1
                # print(ic)
                #####
            np.save("gadf_" + ox + ".npy", np_a)  # gadf_Close.npy

            for i in range(0, x_ma25_2.shape[0] - 50, 1):  # 每100根K畫一張圖
                x = x_ma25_2[i:i + 50]
                x = x.reshape(1, -1)
                x = x * 255

                gadf = GramianAngularField(image_size=24, method='difference')
                x_gadf = gadf.fit_transform(x)

                x_gadf[0].shape  # 24*24=576
                # plt.imshow(x_gadf[0])
                a = np.reshape(x_gadf[0], (1, -1))
                # a.shape
                np_a[i] = a
                ic = ic + 1
                # print(ic)
                #####
            np.save("gadf_ma25.npy", np_a)  # gadf_ma25.npy

            for i in range(0, x_ma100_2.shape[0] - 50, 1):  # 每100根K畫一張圖
                x = x_ma100_2[i:i + 50]
                x = x.reshape(1, -1)
                x = x * 255

                gadf = GramianAngularField(image_size=24, method='difference')
                x_gadf = gadf.fit_transform(x)

                x_gadf[0].shape  # 24*24=576
                # plt.imshow(x_gadf[0])
                a = np.reshape(x_gadf[0], (1, -1))
                # a.shape
                np_a[i] = a
                ic = ic + 1
                # print(ic)
                #####
            np.save("gadf_ma100.npy", np_a)  # gadf_ma100.npy

            for i in range(0, x_atr_2.shape[0] - 50, 1):  # 每100根K畫一張圖
                x = x_atr_2[i:i + 50]
                x = x.reshape(1, -1)
                x = x * 255

                gadf = GramianAngularField(image_size=24, method='difference')
                x_gadf = gadf.fit_transform(x)

                x_gadf[0].shape  # 24*24=576
                # plt.imshow(x_gadf[0])
                a = np.reshape(x_gadf[0], (1, -1))
                # a.shape
                np_a[i] = a
                ic = ic + 1
                # print(ic)
                #####
            np.save("gadf_atr.npy", np_a)  # gadf_atr.npy

            for i in range(0, x_adx_2.shape[0] - 50, 1):  # 每100根K畫一張圖
                x = x_adx_2[i:i + 50]
                x = x.reshape(1, -1)
                x = x * 255

                gadf = GramianAngularField(image_size=24, method='difference')
                x_gadf = gadf.fit_transform(x)

                x_gadf[0].shape  # 24*24=576
                # plt.imshow(x_gadf[0])
                a = np.reshape(x_gadf[0], (1, -1))
                # a.shape
                np_a[i] = a
                #####
            np.save("gadf_adx.npy", np_a)  # gadf_adx.npy

            np.save("y_10000.npy", y1[0:20000])

            arr_c3 = np.load("gadf_Close.npy")
            arr_ma25 = np.load("gadf_ma25.npy")
            arr_ma100 = np.load("gadf_ma100.npy")
            arr_atr = np.load("gadf_atr.npy")
            arr_adx = np.load("gadf_adx.npy")
            y_2d = np.load("y_10000.npy")

            arr_c3.shape, y_2d.shape

            arr_c3 = np.reshape(arr_c3, (-1, 24, 24))
            arr_ma25 = np.reshape(arr_ma25, (-1, 24, 24))
            arr_ma100 = np.reshape(arr_ma100, (-1, 24, 24))
            arr_atr = np.reshape(arr_atr, (-1, 24, 24))
            arr_adx = np.reshape(arr_adx, (-1, 24, 24))

            x_c3 = np.zeros((20000, 5, 24, 24))  # 9995-5,24,24
            xc3 = 0
            for ci in range(0, arr_c3.shape[0]):
                x_c3[ci] = np.array(
                    [arr_c3[ci], arr_ma25[ci], arr_ma100[ci], arr_atr[ci],
                     arr_adx[ci]])
                # print(xc3)
                xc3 += 1
            x_c3.shape  # 9995*5*24*24

            y_2d.shape  # 10000,1    #0:9999
            y_2d.shape[0]
            # set y=5-9999
            y_2d = np.reshape(y_2d[:], (-1, 1))
            y_2d.shape  # 10000:1

            # 成功堆疊5組 Close chart
            x_c3 = np.transpose(x_c3, (0, 2, 3, 1))
            x_c3.shape, y_2d.shape

            x = x_c3
            y = y_2d
            x.shape, y.shape

            # x_再取樣
            x.shape[0]
            x.shape  # (10000, 24, 24, 5)

            x_a = np.zeros((10000, 24, 24, 5))  # 8個矩陣 576*8=4608
            xc = 0
            for xs in range(0, x.shape[0], 2):
                x_a[xc] = x[xs]
                xc = xc + 1
                # print(xc)
            # y_再取樣
            y_2d.shape
            y_2d_a = np.zeros((10000,))
            y_2d_a.shape
            xc = 0
            for ys in range(0, y_2d.shape[0], 2):
                y_2d_a[xc] = y_2d[ys, 0]
                # print(xc)
                xc += 1
            x_a.shape, y_2d_a.shape
            x = x_a
            y = y_2d_a.reshape(-1, 1)
            x.shape, y.shape
            # 5454 array
            np.save("5000_x.npy", x)
            np.save("5000_y.npy", y)

            x1 = np.load("5000_x.npy")
            y = np.load("5000_y.npy")
            y_OneHot = np_utils.to_categorical(y.astype('int'))
            y1 = y_OneHot
            x1.shape, y1.shape  # 4998

            n_split = 10

            cc = 0
            for train_index, test_index in KFold(n_split).split(x1):
                x_train, x_test = x1[train_index], x1[test_index]
                y_train, y_test = y1[train_index], y1[test_index]
                x_train.shape, y_train.shape, x_test.shape, y_test.shape
                model = cnn()
                # model.fit(x_train, y_train, epochs=20)
                my_callbacks = [
                    EarlyStopping(monitor='val_loss', patience=25, verbose=2,
                                  mode='auto')]

                md = model.fit(x=x_train, y=y_train,
                               validation_data=(x_test, y_test), epochs=50,
                               batch_size=30, verbose=2, callbacks=my_callbacks)
                scores = model.evaluate(x_test, y_test)
                print('Model evaluation ', scores[1] * 100, "%")
                locals()['cvscores_%s' % (ii)][ai, cc] = scores[1] * 100
                cc += 1
            # print(np.round(locals()['cvscores_%s' % (ii)][ai], 2))  # 印出CV的各次百分比
            # print("%.2f%% (+/- %.2f%%)" % (
            # np.mean(locals()['cvscores_%s' % (ii)][ai]), np.std(locals()['cvscores_%s' % (ii)][ai])))  # 印出平均值（標準差）
            locals()['cvscores_%s' % (ii)][ai, 10] = np.mean(
                locals()['cvscores_%s' % (ii)][ai, 0:9])
            locals()['cvscores_%s' % (ii)][ai, 11] = np.std(
                locals()['cvscores_%s' % (ii)][ai, 0:9])
            fn = "cvscores_" + str(ii) + ".npy"
            np.save(fn, locals()['cvscores_%s' % (ii)])
            lrs=str(lr)
            fn1 = "cvscores_" + str(ii) +"+"+str(lrs.strip("."))+".csv"
            np.savetxt(fn1, locals()['cvscores_%s' % (ii)], delimiter=",")

# def for_loop
#     for ii in range(0, 4):
#         lr = 0.0002 * ii
#         # locals()['cvscores_%s' % (ii)]
#         locals()['cvscores_%s' % (ii)] = np.zeros((10, 12))
#         for ai in range(0, 8):
#             def cnn():
#                 model = Sequential()
#                 # CNN1
#                 model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
#                                  input_shape=(24, 24, 5), activation='relu'))
#                 # MaxPooling1
#                 model.add(MaxPool2D(pool_size=(2, 2)))
#                 # CNN2
#                 model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
#                                  activation='relu'))
#                 # MaxPooling2
#                 model.add(MaxPool2D(pool_size=(2, 2)))
#                 # CNN3
#                 model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same',
#                                  activation='relu'))
#                 # dropout 避免Overfitting
#                 model.add(Dropout(0.4))
#                 # 平坦化
#                 model.add(Flatten())
#                 model.add(Dense(1024, activation='relu'))
#                 model.add(Dropout(0.4))
#                 model.add(Dense(256, activation='relu'))
#                 model.add(Dense(64, activation='relu'))
#                 model.add(Dense(8, activation='relu'))
#                 model.add(Dense(2, activation='sigmoid'))
#                 # 模型訓練
#                 adam = keras.optimizers.Adam(lr=lr, beta_1=0.9,
#                                              beta_2=0.999,
#                                              epsilon=None, decay=0.0,
#                                              amsgrad=True)
#                 model.compile(loss='binary_crossentropy', optimizer=adam,
#                               metrics=['accuracy'])
#                 return model
#
#             xi = ai
#             # total 的x 為 df_x.npy    y為 y_pre.ngy
#             x_all = np.load("df_x.npy")
#             y_all = np.load("y_pre.npy")
#             x_all.shape, y_all.shape  # x=17,36620  y=36620,
#
#             x = np.copy(x_all[3, :])  # 第3欄是收盤價
#             x_ma25 = np.copy(x_all[5, :])
#             x_ma100 = np.copy(x_all[6, :])
#             x_atr = np.copy(x_all[10, :])
#             x_adx = np.copy(x_all[16, :])
#
#             y = y_all
#
#             # 每2000一STEP 例如 train 0:10000 test 2000-12000
#
#             xj = xi * 2000  # 間隔
#             x1 = x[xj:xj + 20050]  # 100根，做圖i~~i+100
#             x_ma25_1 = x_ma25[xj:xj + 20050]
#             x_ma100_1 = x_ma100[xj:xj + 20050]
#             x_atr_1 = x_atr[xj:xj + 20050]
#             x_adx_1 = x_adx[xj:xj + 20050]
#
#             y1 = y[xj:xj + 20050]
#
#             x1.shape, y1.shape
#             x1 = np.reshape(x1, (1, -1))
#             normalizer = Normalizer().fit(x1)
#             x2 = normalizer.transform(x1)
#
#             x_ma25_1 = np.reshape(x_ma25_1, (1, -1))
#             normalizer = Normalizer().fit(x_ma25_1)
#             x_ma25_2 = normalizer.transform(x_ma25_1)
#
#             x_ma100_1 = np.reshape(x_ma100_1, (1, -1))
#             normalizer = Normalizer().fit(x_ma100_1)
#             x_ma100_2 = normalizer.transform(x_ma100_1)
#
#             x_atr_1 = np.reshape(x_atr_1, (1, -1))
#             normalizer = Normalizer().fit(x_atr_1)
#             x_atr_2 = normalizer.transform(x_atr_1)
#
#             x_adx_1 = np.reshape(x_adx_1, (1, -1))
#             normalizer = Normalizer().fit(x_adx_1)
#             x_adx_2 = normalizer.transform(x_adx_1)
#
#             x2 = np.ravel(x2.T)  # 再將X2轉置，並展開RAVEL
#             x_ma25_2 = np.ravel(x_ma25_2.T)
#             x_ma100_2 = np.ravel(x_ma100_2.T)
#             x_atr_2 = np.ravel(x_atr_2.T)
#             x_adx_2 = np.ravel(x_adx_2.T)
#
#             x2.shape, x_ma25_2.shape, x_ma100_2.shape, x_atr_2.shape, x_adx_2.shape, y1.shape
#
#             ox = "Close"  # 存npy 時的命名：若為"ma"則檔名為 gadf_ma.npy
#             ic = 0  # i的 count
#             np_a = np.zeros((20000, 576))
#             x2.shape
#
#             for i in range(0, x2.shape[0] - 50, 1):  # 每100根K畫一張圖
#                 x = x2[i:i + 50]
#                 x = x.reshape(1, -1)
#                 x = x * 255
#
#                 gadf = GramianAngularField(image_size=24, method='difference')
#                 x_gadf = gadf.fit_transform(x)
#
#                 x_gadf[0].shape  # 24*24=576
#                 # plt.imshow(x_gadf[0])
#                 a = np.reshape(x_gadf[0], (1, -1))
#                 # a.shape
#                 np_a[i] = a
#                 ic = ic + 1
#                 # print(ic)
#                 #####
#             np.save("gadf_" + ox + ".npy", np_a)  # gadf_Close.npy
#
#             for i in range(0, x_ma25_2.shape[0] - 50, 1):  # 每100根K畫一張圖
#                 x = x_ma25_2[i:i + 50]
#                 x = x.reshape(1, -1)
#                 x = x * 255
#
#                 gadf = GramianAngularField(image_size=24, method='difference')
#                 x_gadf = gadf.fit_transform(x)
#
#                 x_gadf[0].shape  # 24*24=576
#                 # plt.imshow(x_gadf[0])
#                 a = np.reshape(x_gadf[0], (1, -1))
#                 # a.shape
#                 np_a[i] = a
#                 ic = ic + 1
#                 # print(ic)
#                 #####
#             np.save("gadf_ma25.npy", np_a)  # gadf_ma25.npy
#
#             for i in range(0, x_ma100_2.shape[0] - 50, 1):  # 每100根K畫一張圖
#                 x = x_ma100_2[i:i + 50]
#                 x = x.reshape(1, -1)
#                 x = x * 255
#
#                 gadf = GramianAngularField(image_size=24, method='difference')
#                 x_gadf = gadf.fit_transform(x)
#
#                 x_gadf[0].shape  # 24*24=576
#                 # plt.imshow(x_gadf[0])
#                 a = np.reshape(x_gadf[0], (1, -1))
#                 # a.shape
#                 np_a[i] = a
#                 ic = ic + 1
#                 # print(ic)
#                 #####
#             np.save("gadf_ma100.npy", np_a)  # gadf_ma100.npy
#
#             for i in range(0, x_atr_2.shape[0] - 50, 1):  # 每100根K畫一張圖
#                 x = x_atr_2[i:i + 50]
#                 x = x.reshape(1, -1)
#                 x = x * 255
#
#                 gadf = GramianAngularField(image_size=24, method='difference')
#                 x_gadf = gadf.fit_transform(x)
#
#                 x_gadf[0].shape  # 24*24=576
#                 # plt.imshow(x_gadf[0])
#                 a = np.reshape(x_gadf[0], (1, -1))
#                 # a.shape
#                 np_a[i] = a
#                 ic = ic + 1
#                 # print(ic)
#                 #####
#             np.save("gadf_atr.npy", np_a)  # gadf_atr.npy
#
#             for i in range(0, x_adx_2.shape[0] - 50, 1):  # 每100根K畫一張圖
#                 x = x_adx_2[i:i + 50]
#                 x = x.reshape(1, -1)
#                 x = x * 255
#
#                 gadf = GramianAngularField(image_size=24, method='difference')
#                 x_gadf = gadf.fit_transform(x)
#
#                 x_gadf[0].shape  # 24*24=576
#                 # plt.imshow(x_gadf[0])
#                 a = np.reshape(x_gadf[0], (1, -1))
#                 # a.shape
#                 np_a[i] = a
#                 #####
#             np.save("gadf_adx.npy", np_a)  # gadf_adx.npy
#
#             np.save("y_10000.npy", y1[0:20000])
#
#             arr_c3 = np.load("gadf_Close.npy")
#             arr_ma25 = np.load("gadf_ma25.npy")
#             arr_ma100 = np.load("gadf_ma100.npy")
#             arr_atr = np.load("gadf_atr.npy")
#             arr_adx = np.load("gadf_adx.npy")
#             y_2d = np.load("y_10000.npy")
#
#             arr_c3.shape, y_2d.shape
#
#             arr_c3 = np.reshape(arr_c3, (-1, 24, 24))
#             arr_ma25 = np.reshape(arr_ma25, (-1, 24, 24))
#             arr_ma100 = np.reshape(arr_ma100, (-1, 24, 24))
#             arr_atr = np.reshape(arr_atr, (-1, 24, 24))
#             arr_adx = np.reshape(arr_adx, (-1, 24, 24))
#
#             x_c3 = np.zeros((20000, 5, 24, 24))  # 9995-5,24,24
#             xc3 = 0
#             for ci in range(0, arr_c3.shape[0]):
#                 x_c3[ci] = np.array(
#                     [arr_c3[ci], arr_ma25[ci], arr_ma100[ci], arr_atr[ci],
#                      arr_adx[ci]])
#                 # print(xc3)
#                 xc3 += 1
#             x_c3.shape  # 9995*5*24*24
#
#             y_2d.shape  # 10000,1    #0:9999
#             y_2d.shape[0]
#             # set y=5-9999
#             y_2d = np.reshape(y_2d[:], (-1, 1))
#             y_2d.shape  # 10000:1
#
#             # 成功堆疊5組 Close chart
#             x_c3 = np.transpose(x_c3, (0, 2, 3, 1))
#             x_c3.shape, y_2d.shape
#
#             x = x_c3
#             y = y_2d
#             x.shape, y.shape
#
#             # x_再取樣
#             x.shape[0]
#             x.shape  # (10000, 24, 24, 5)
#
#             x_a = np.zeros((10000, 24, 24, 5))  # 8個矩陣 576*8=4608
#             xc = 0
#             for xs in range(0, x.shape[0], 2):
#                 x_a[xc] = x[xs]
#                 xc = xc + 1
#                 # print(xc)
#             # y_再取樣
#             y_2d.shape
#             y_2d_a = np.zeros((10000,))
#             y_2d_a.shape
#             xc = 0
#             for ys in range(0, y_2d.shape[0], 2):
#                 y_2d_a[xc] = y_2d[ys, 0]
#                 # print(xc)
#                 xc += 1
#             x_a.shape, y_2d_a.shape
#             x = x_a
#             y = y_2d_a.reshape(-1, 1)
#             x.shape, y.shape
#             # 5454 array
#             np.save("5000_x.npy", x)
#             np.save("5000_y.npy", y)
#
#             x1 = np.load("5000_x.npy")
#             y = np.load("5000_y.npy")
#             y_OneHot = np_utils.to_categorical(y.astype('int'))
#             y1 = y_OneHot
#             x1.shape, y1.shape  # 4998
#
#             n_split = 10
#
#             cc = 0
#             for train_index, test_index in KFold(n_split).split(x1):
#                 x_train, x_test = x1[train_index], x1[test_index]
#                 y_train, y_test = y1[train_index], y1[test_index]
#                 x_train.shape, y_train.shape, x_test.shape, y_test.shape
#                 model = cnn()
#                 # model.fit(x_train, y_train, epochs=20)
#                 my_callbacks = [
#                     EarlyStopping(monitor='val_loss', patience=35, verbose=2,
#                                   mode='auto')]
#
#                 md = model.fit(x=x_train, y=y_train,
#                                validation_data=(x_test, y_test), epochs=45,
#                                batch_size=30, verbose=2, callbacks=my_callbacks)
#                 scores = model.evaluate(x_test, y_test)
#                 print('Model evaluation ', scores[1] * 100, "%")
#                 locals()['cvscores_%s' % (ii)][ai, cc] = scores[1] * 100
#                 cc += 1
#             # print(np.round(locals()['cvscores_%s' % (ii)][ai], 2))  # 印出CV的各次百分比
#             # print("%.2f%% (+/- %.2f%%)" % (
#             # np.mean(locals()['cvscores_%s' % (ii)][ai]), np.std(locals()['cvscores_%s' % (ii)][ai])))  # 印出平均值（標準差）
#             locals()['cvscores_%s' % (ii)][ai, 10] = np.mean(
#                 locals()['cvscores_%s' % (ii)][ai, 0:5])
#             locals()['cvscores_%s' % (ii)][ai, 11] = np.std(
#                 locals()['cvscores_%s' % (ii)][ai, 0:5])
#             fn = "cvscores_" + str(ii) + ".npy"
#             np.save(fn, locals()['cvscores_%s' % (ii)])
#             fn1 = "cvscores_" + str(ii) + ".csv"
#             np.savetxt(fn1, locals()['cvscores_%s' % (ii)], delimiter=",")


