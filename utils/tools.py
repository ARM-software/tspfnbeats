import os, shutil, yaml, logging

# folder to load config file
CONFIG_PATH = "../config/"


# Function to load yaml configuration file
def load_config(config_name: str) -> dict:
    if not os.path.isfile(config_name):
        config_name = os.path.join(CONFIG_PATH, config_name)
    assert os.path.isfile(config_name), f'{config_name} could not be found'

    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


def smape_simple(y_true, y_pred, w=1.0) -> float:
    import tensorflow as tf
    values = 200.0 * tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + tf.math.abs(y_pred) + 1.0e-5)
    _loss = tf.reduce_mean(tf.reduce_mean(w * values, axis=1), axis=0)
    return float(_loss)


def mk_clear_dir(d: str, delete_existing: bool) -> str:
    if os.path.exists(d) and delete_existing:
        shutil.rmtree(d)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def silence_tensorflow():
    """
        Imported from: https://github.com/LucaCappelletti94/silence_tensorflow
    :return:
    """
    """Silence every warning of notice from tensorflow."""
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)