import argparse
import yaml
from utils import *
from trapdoor_enabled_defense import trapdoor_enabled_defense

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='batch run the trapdoor task for the given dataset')
    parser.add_argument('--config', dest='config', default='configs/mnist.yaml')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--distinct_method', dest='distinct_method', default="orthogonal")
    args = parser.parse_args()
    with open(args.config) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if args.device == -1:
        args.device = params['device']
    params['distinct_method'] = args.distinct_method
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    fix_gpu_memory()
    fix_random_seed(params['random_seed'])
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=150)
    gen_train, gen_test = load_data(params["dataset_info"], params["training_setting"]["batch_size"])
    trapdoor_enabled_defense(gen_train, gen_test, **params)
