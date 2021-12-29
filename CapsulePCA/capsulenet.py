"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python CapsNet.py
       python CapsNet.py --epochs 100
       python CapsNet.py --epochs 100 --num_routing 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='elu', name='conv1')(x)
    batchn = layers.BatchNormalization()(conv1)
    
    conv2 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='same', activation='elu', name='conv2')(batchn)
    
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)

    
    out_caps = Length(name='capsnet')(digitcaps)

    
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  
    masked = Mask()(digitcaps)  

    
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='elu', input_dim=16*n_class))
    decoder.add(layers.Dense(784, activation='elu'))
    decoder.add(layers.Dropout(0.3))                         
    decoder.add(layers.Dense(1024, activation='elu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.95 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction,
                                           horizontal_flip=True)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()


def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist, fashion_mnist
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_sc_train = scaler.transform(x_train)
    X_sc_test = scaler.transform(x_test)
    
    pca = PCA(n_components=324)
    X_pca_train = pca.fit_transform(X_sc_train)
    X_pca_test = pca.transform(X_sc_test)
    
    
    inv_pca = pca.inverse_transform(X_pca_train)
    inv_sc = scaler.inverse_transform(inv_pca)
    
    inv_pca_test = pca.inverse_transform(X_pca_test)
    inv_sc_test = scaler.inverse_transform(inv_pca_test)
    
    x_train = inv_sc.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = inv_sc_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    
    # x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    
    # x_train = x_train.reshape(60000, 784, 1)
    # x_test = x_test.reshape(60000, 784, 1)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=16)
    # x_train = pca.fit(x_train).transform(x_train)
    # x_test = pca.fit(x_test).transform(x_test)

    # x_train = x_train.reshape(-1, 4, 4, 1)
    # x_test = x_test.reshape(-1, 4, 4, 1)
    
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=65, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                num_routing=args.num_routing)
    model.summary()
    #model.load_weights('C:/Users/Ehsan/Downloads/trained_model (1).h5')
    # img = cv2.imread('E:/seven.png')
    # img = cv2.resize(img,(28, 28))
    # img = cv2.reshape(img, [28, 28, 1])
    
    
    #test(model=eval_model, data=(x_test, y_test))

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test))

