import yaml

class Config():
    def __init__(self,episodes=5000,tests=10,test_period=100,learning_rate=0.001,deviation=0.03,discount_rate=0.95):
        self.episodes = episodes
        self.tests = tests
        self.test_period = test_period
        self.learning_rate = learning_rate
        self.deviation = deviation
        self.discount_rate = discount_rate

def to_dict(cfg):
    d = {'episodes':cfg.episodes,
        'tests':cfg.tests,
        'test_period':cfg.test_period,
        'learning_rate':cfg.learning_rate,
        'deviation':cfg.deviation,
        'discount_rate':cfg.discount_rate}
    return d

def to_cfg(d):
    cfg = Config(episodes=int(d['episodes']),
                tests=int(d['tests']),
                test_period=int(d['test_period']),
                learning_rate=float(d['learning_rate']),
                deviation=float(d['deviation']),
                discount_rate=float(d['discount_rate']))
    return cfg

def write_yaml(d,file='config.yaml'):
    with open(file,'w') as f:
        yaml.dump(d,f)

def read_yaml(file):
    with open(file) as f:
        out = yaml.load(f,Loader=yaml.FullLoader)
        return out

def scale(target,max,min=0):
    if min < 0:
        target -= min
        max -= min
    elif min > 0:
        target += min
        max += min
    target = target / max
    return target