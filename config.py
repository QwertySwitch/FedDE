############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
num_epochs = 10
batch_size = 128
optim_type = 'SGD'

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'fmnist': (0.5,),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'fmnist': (0.5,),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 200):
        optim_factor = 3
    elif(epoch > 160):
        optim_factor = 2
    elif(epoch > 80):
        optim_factor = 1

    return init*math.pow(0.1, optim_factor)

def learning_rate2(init, epoch):
    optim_factor = 0
    if(epoch > 80):
        optim_factor = 3
    elif(epoch > 50):
        optim_factor = 2
    elif(epoch > 30):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
