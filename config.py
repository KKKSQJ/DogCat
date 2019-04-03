#coding:utf-8
import warnings
class DefaultConfig(object):
    env = 'default' # visdom环境
    model = 'ResNet34' #使用的模型，名字必须和models/__init__.py中的名字一样

    train_data_root = './data/train/'
    test_data_root = './data/test1/'
    load_model_path = None #加载预训练模型的路径，none表示不加载

    batch_size = 32
    use_gpu = True
    num_workers = 4
    print_freq = 20

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001
    lr_decay = 0.5 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5

def parse(self,kwargs):
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k.v)

    print('user config:')
    for k,v in self.__class__.__dict__.items():
        if not k.startwith("__"):
            print(k,getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()
