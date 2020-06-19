import tensorflow_datasets as tfds

def process(arr, covnet):
    arr_size = arr.shape[1]
    if covnet:
        arr_shape = (-1, arr_size, arr_size, arr_size.shape[1])
    else:
        arr_shape =(-1, arr_size*arr_size)
    return (arr / 128.0 - 1).reshape(arr_shape)

def load_data(datast, train_size, test_size, covnet = False):
    (train_images, train_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load(
        datast, split=['train', 'test'], batch_size=-1, as_supervised=True))
    train_images = process(train_images, covnet)
    test_images = process(test_images, covnet)
    train_images = train_images[:train_size]
    train_labels = train_labels[:train_size]
    test_images = test_images[:test_size]
    test_labels = test_labels[:test_size]
    train_labels = 2 * (train_labels % 2 == 0) - 1
    test_labels = 2 * (test_labels % 2 == 0) - 1
    return train_images, train_labels, test_images, test_labels
