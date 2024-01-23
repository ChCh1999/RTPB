from omegaconf import OmegaConf, ListConfig, DictConfig

DEFAULT_CONFIG_FILE = "configs/default.yaml"
default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)


def set_config(cfg):
    global default_config
    default_config = cfg


def get_config():
    return default_config


def check_config(cfg):
    cfg.LOG_WINDOW_SIZE = 320 // cfg.SOLVER.IMS_PER_BATCH
    cfg = leaf_to_container(cfg)[0]
    return cfg


def leaf_to_container(node):
    if isinstance(node, DictConfig):
        is_leaf = False
        for k, v in node.items():
            node[k], _ = leaf_to_container(v)
    elif isinstance(node, ListConfig):
        is_leaf = False
        leaf_flags = [False for _ in range(len(node))]
        for i, n in enumerate(node):
            node[i], leaf_flags[i] = leaf_to_container(n)
        if all(leaf_flags):
            node = OmegaConf.to_container(node)
    else:
        is_leaf = True
    return node, is_leaf


def flatten_conf(cfg, prefix="", delimiter="."):
    cfg = OmegaConf.to_object(cfg)

    def extract(cfg, prefix):
        cfg_flatten = {}
        for k, v in cfg.items():
            if prefix:
                prefix_k = delimiter.join([prefix, k])
            else:
                prefix_k = k
            if isinstance(v, dict):
                cfg_flatten.update(extract(v, prefix_k))
            else:
                cfg_flatten[prefix_k] = v
        return cfg_flatten

    return extract(cfg, prefix)
