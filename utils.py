import yaml
import os

def get_config(filename='./config.yml'):
    conf_file = open(filename, 'r')
    return yaml.load(conf_file)

def set_env():
    conf = get_config()
    if 'env' in conf:
        env = conf['env']
        os.environ.setdefault('CUDA_ROOT', env['cuda_root'])
        os.environ.setdefault('THEANO_FLAGS', env['theano_flags'])

        cuda_lib = env.get('lib_path', '')
        initial_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = cuda_lib
        if initial_path:
            os.environ['LD_LIBRARY_PATH'] += ":" + initial_path

