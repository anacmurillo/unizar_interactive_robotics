import numpy as np
import skimage.io
import skimage.transform
import cPickle
import gzip
from Scene.FMI import *

actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10']

def rescale(img, input_height, input_width):
    aspect = img.shape[1] / float(img.shape[0])
    if aspect > 1:
        return skimage.transform.resize(img, (input_width, int(aspect * input_height)))
    elif aspect < 1:
        return skimage.transform.resize(img, (int(input_width / aspect), input_height))
    else:
        return skimage.transform.resize(img, (input_width, input_height))


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def prepare_image(img):
    # img = skimage.io.imread(img_path)
    img = skimage.img_as_float(img)
    img = rescale(img, 224, 224)
    img = crop_center(img, 224, 224)
    img = img.swapaxes(1, 2).swapaxes(0, 1)  # HWC to CHW dimension
    img = img[(2, 1, 0), :, :]  # RGB to BGR color order
    img = img * 255 - 128  # Subtract mean = 128
    return img.astype(np.float32)


import os, glob, random


def make_batch(iterable, batch_size=1):
    length = len(iterable)
    for index in range(0, length, batch_size):
        yield iterable[index:min(index + batch_size, length)]


class DogsCatsDataset(object):
    """ Dogs and cats dataset reader """

    def __init__(self, split="train",Bool=True,batch_size=1):
        path = '/home/pazagra/Data2/'
        if Bool:
            users = ["user" + n.__str__() for n in xrange(1, 11) if "user" + n.__str__() == split]
        else:
            users = ["user" + n.__str__() for n in xrange(1, 11) if "user" + n.__str__() != split ]
        self.data = []
        self.labe = []
        self.Bool = Bool
        self.image_files = []
        self.labels = []
        for user in users:
            # print user_c + "_Train_"+user
            for action in actions:
                files = gzip.open(path + "/" + user +'_'+action+ ".gpz", 'r')
                FM = cPickle.load(files)
                files.close()
                if FM.Candidate_patch is None:
                    print user + " Has no Candidate"
                    continue
                for candidate in FM.Candidate_patch:
                    if candidate.Label is not None:
                        self.data.append((candidate.patch, candidate.Label))
                        # print candidate.patch.shape
                        # print type(candidate.Label)==list
                        self.labe.append(candidate.Label)
        self.categories = {}
        i = 0
        # print self.labe
        for l in np.unique(np.array(self.labe)):
            self.categories[l] = i
            i += 1
        # self.categories = {"dog": 0, "cat": 1}
        for im, lab in self.data:
            self.image_files.append(im)
            self.labels.append(self.categories.get(lab, -1))
        ina = self.__len__()%batch_size
        self.image_files=self.image_files[0:self.__len__()-ina]
        self.labels = self.labels[0:self.__len__()-ina]
        # print self.categories
        # print max(self.labels)
        # self.image_files = list(glob.glob(os.path.join(data_dir, split, "*.jpg")))
        # self.labels = [self.categories.get(os.path.basename(path).strip().split(".")[0], -1)
        #               for path in self.image_files]

    def __getitem__(self, index):
        image = prepare_image(self.image_files[index])
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)

    def read(self, batch_size=1, shuffle=True):
        """Read (image, label) pairs in batch"""
        order = list(range(len(self)))
        if shuffle:
            random.shuffle(order)
        for batch in make_batch(order, batch_size):
            images, labels, images1 = [], [], []
            for index in batch:
                image, label = self[index]
                images.append(image)
                labels.append(label)
            yield np.stack(images).astype(np.float32), np.stack(labels).astype(np.int32).reshape((batch_size,))


from caffe2.python import core, workspace, model_helper, optimizer, brew
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.proto import caffe2_pb2

PREDICT_NET = "/home/pazagra/caffe2det/pytorch/build/caffe2/python/models/vgg16/predict_net.pb"
INIT_NET = "/home/pazagra/caffe2det/pytorch/build/caffe2/python/models/vgg16/init_net.pb"


def AddPredictNet(model, predict_net_path):
    predict_net_proto = caffe2_pb2.NetDef()
    with open(predict_net_path, "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    # for op in predict_net_proto.op:
    #	print op.output[0]
    model.net = core.Net(predict_net_proto)
    # Fix dimension incompatibility
    # model.Squeeze("z", "softmax", dims=[2, 3])


def AddInitNet(model, init_net_path, out_dim=22, params_to_learn=None):
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_path, "rb") as f:
        init_net_proto.ParseFromString(f.read())

    # Define params to learn in the model.
    for op in init_net_proto.op:
        param_name = op.output[0]
        if params_to_learn is None or op.output[0] in params_to_learn:
            tags = (ParameterTags.WEIGHT if param_name.endswith("_w")
                    else ParameterTags.BIAS)
            model.create_param(
                param_name=param_name,
                shape=op.arg[0],
                initializer=initializers.ExternalInitializer(),
                tags=tags,
            )
    # for op in init_net_proto.op:
    #	print op.output[0]
    #	print op.arg[0].name

    # Remove conv10_w, conv10_b initializers at (50, 51)
    init_net_proto.op.pop(14)
    init_net_proto.op.pop(15)

    # Add new initializers for conv10_w, conv10_b
    model.param_init_net = core.Net(init_net_proto)
    model.param_init_net.XavierFill([], "gpu_0/fc8_w", shape=[out_dim, 1024])
    model.param_init_net.ConstantFill([], "gpu_0/fc8_b", shape=[out_dim])


def finetune(model, out_dim):
    model.param_init_net.XavierFill([], "gpu_0/fc8_w", shape=[out_dim, 1024])
    model.param_init_net.ConstantFill([], "gpu_0/fc8_b", shape=[out_dim])


def AddTrainingOperators(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], "xent")
    loss = model.AveragedLoss(xent, "loss")
    brew.accuracy(model, [softmax, label], "accuracy")
    model.AddGradientOperators([loss])
    opt = optimizer.build_sgd(model, base_learning_rate=0.01)
    for param in model.GetOptimizationParamInfo():
        opt(model.net, model.param_init_net, param)


train_model = model_helper.ModelHelper("train_net")
AddPredictNet(train_model, PREDICT_NET)
AddInitNet(train_model, INIT_NET, params_to_learn=["gpu_0/fc7_w", "gpu_0/fc7_b","gpu_0/fc8_w", "gpu_0/fc8_b"])  # Use None to learn everything.
AddTrainingOperators(train_model, "gpu_0/softmax", "gpu_0/label")


def SetDeviceOption(model, device_option):
    # Clear op-specific device options and set global device option.
    for net in ("net", "param_init_net"):
        net_def = getattr(model, net).Proto()
        net_def.device_option.CopyFrom(device_option)
        for op in net_def.op:
            # Some operators are CPU-only.
            if op.output[0] not in ("optimizer_iteration", "iteration_mutex"):
                op.ClearField("device_option")
                op.ClearField("engine")
        setattr(model, net, core.Net(net_def))

train_dataset = DogsCatsDataset('user1',True,10)

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.CUDA
device_option.cuda_gpu_id = 0
SetDeviceOption(train_model, device_option)
users = ["user" + n.__str__() for n in xrange(1, 11)]
accuracy2 =[]
for user in users:
    workspace.ResetWorkspace()

    # Initialization.
    train_dataset = DogsCatsDataset(user,False,10)
    for image, label in train_dataset.read(batch_size=1):
        workspace.FeedBlob("gpu_0/data", image, device_option=device_option)
        workspace.FeedBlob("gpu_0/label", label, device_option=device_option)
        break
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    # Main loop.
    batch_size = 10
    print_freq = 10
    losses = []
    for epoch in range(5):
        for index, (image, label) in enumerate(train_dataset.read(batch_size)):
            workspace.FeedBlob("gpu_0/data", image, device_option=device_option)
            workspace.FeedBlob("gpu_0/label", label, device_option=device_option)
            workspace.RunNet(train_model.net)
            accuracy = float(workspace.FetchBlob("accuracy"))
            loss = workspace.FetchBlob("loss").mean()
            losses.append(loss)
            if index % print_freq == 0:
                print("[{}][{}/{}] loss={}, accuracy={}".format(
                    epoch, index, int(len(train_dataset) / batch_size),
                    loss, accuracy))

    import matplotlib.pyplot as plt

    plt.plot(losses)


    deploy_model = model_helper.ModelHelper("deploy_net")
    AddPredictNet(deploy_model, PREDICT_NET)
    SetDeviceOption(deploy_model, device_option)

    import cv2
    a=[]
    batch_size = 1
    test_dataset = DogsCatsDataset(user,True,1)
    for index, (image, label) in enumerate(test_dataset.read(batch_size)):
        image = image[np.newaxis, :]
        workspace.FeedBlob("data", image, device_option=device_option)
        workspace.RunNetOnce(deploy_model.net)
        result = workspace.FetchBlob("gpu_0/softmax")[0]
        # print image1.shape
        # print image1.dtype
        # cv2.imshow(" ",image1)
        # cv2.waitKey(0)
        # print label
        maximo = max(result)
        i, = np.where(result == maximo)
        for k in train_dataset.categories.keys():
            if train_dataset.categories[k]==i:
                output =k
        for k in test_dataset.categories.keys():
            if test_dataset.categories[k]==label:
                labele= k
        a.append(output==labele)
    if len(np.unique(a,return_counts=True)[1]) ==1:
        accuracy2.append(0.0)
    else:
        accuracy2.append(float(np.unique(a,return_counts=True)[1][1])/float(sum(np.unique(a,return_counts=True)[1])))
print accuracy2
plt.xlabel("iterations")
plt.ylabel("loss/accuracy")
plt.grid("on")

plt.show()
