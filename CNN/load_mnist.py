import numpy as np
import struct
#import numba as nb

from  sklearn.preprocessing import  OneHotEncoder

# 训练集文件
train_images_idx3_ubyte_file = 'MNIST_data_/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'MNIST_data_/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 'MNIST_data_/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'MNIST_data_/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_mnist_data():
    x_test=decode_idx3_ubyte(test_images_idx3_ubyte_file)
    x_train=decode_idx3_ubyte(train_images_idx3_ubyte_file)
    y_train=decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    y_test=decode_idx1_ubyte(test_labels_idx1_ubyte_file)

    x_train=x_train.reshape((-1,28*28,1))
    x_test=x_test.reshape((-1,28*28,1))

    y_test=y_test.reshape((-1,1))
    y_train=y_train.reshape((-1,1))

    ohe = OneHotEncoder()

    y_train = ohe.fit_transform(np.matrix(y_train)).toarray()
    y_test = ohe.fit_transform(np.matrix(y_test)).toarray()

    y_test=y_test.reshape((-1,10,1))
    y_train=y_train.reshape((-1,10,1))

    train_data=[]
    test_data=[]

    for i,(x,y) in enumerate(zip(x_train,y_train)):
        if i%20==0:
            train_data.append((x,y))

    for i,(x,y) in enumerate(zip(x_test,y_test)):
        if i % 10 == 0:
            test_data.append((x,y))

    return train_data,test_data

#@nb.jit()
def load_mnist_img(mini_batch_size):

    x_test=decode_idx3_ubyte(test_images_idx3_ubyte_file)
    x_train=decode_idx3_ubyte(train_images_idx3_ubyte_file)
    y_train=decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    y_test=decode_idx1_ubyte(test_labels_idx1_ubyte_file)

    train_data=[]
    test_data=[]
    #
    for i,(x,y) in enumerate(zip(x_train,y_train)):
        if i%20==0:
            train_data.append([x,y])

    for i,(x,y) in enumerate(zip(x_test,y_test)):
        if i % 10 == 0:
            test_data.append([x,y])


    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    m=len(x_train)

    train_data_random=[]
    test_data_random=[]
    for i in range(0,m,mini_batch_size):

        temp_x=np.zeros((mini_batch_size,28,28))
        temp_y=np.zeros((mini_batch_size))

        for i,(x,y) in enumerate(train_data[i:i+mini_batch_size]):

            temp_x[i] = x
            temp_y[i] = y

        train_data_random.append((temp_x,temp_y))
    m = len(x_test)

    for i in range(0,m,mini_batch_size):

        temp_x=np.zeros((mini_batch_size,28,28))
        temp_y=np.zeros((mini_batch_size))

        for i,(x,y) in enumerate(test_data[i:i+mini_batch_size]):

            temp_x[i] = x
            temp_y[i] = y

        test_data_random.append((temp_x,temp_y))


    return train_data_random,test_data_random


