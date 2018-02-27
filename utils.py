"""
Utility file for reading train/test data of MNIST files (images and their labels).
"""


def bytes2int(str):
    """
    Converts 4 bytes string to int
    :param str: 4 chars string
    :return: Integer of the 4 bytes from the string
    """
    return int(str.encode('hex'), 16)


def char_digit2int(c):
    """
    Converts a char digit to an int with the actual value
    :param c: The char digit
    :return: If char is between '0' to '9' returns the digit, otherwise -1
    """
    if c < '0' or c > '9': return -1
    return ord(c) - ord('0')


def read_images_file(fname):
    """
    Reads an MNIST images file, where its format is 
    :param fname:
    :return:
    """
    imgs_file = open(fname, "rb")
    magic_number = bytes2int(imgs_file.read(4))
    img_num = bytes2int(imgs_file.read(4))
    rows, cols = bytes2int(imgs_file.read(4)), bytes2int(imgs_file.read(4))

    # now reading images
    images = [imgs_file.read(rows * cols) for _ in xrange(img_num)]

    return img_num, (rows, cols), images


def read_labels_file(fname):
    labels_file = open(fname, "rb")
    magic_number = bytes2int(labels_file.read(4))
    label_num = bytes2int(labels_file.read(4))
    labels = [char_digit2int(labels_file.read(1)) for _ in xrange(label_num)]
    return label_num, labels


def read_images_and_labels(imgs_file, labels_file):
    img_num, img_sizes, images = read_images_file(imgs_file)
    label_num, labels = read_labels_file(labels_file)
    if label_num != img_num or img_num != len(images) or label_num != len(labels): return None
    return img_num, img_sizes, images, labels
