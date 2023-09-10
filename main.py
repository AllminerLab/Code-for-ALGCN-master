import yaml
import os
from src.trainer.base_trainer import *
def set_logging():
    logging.basicConfig(level="INFO")


def seed_everything(seed=2020):
    logging.info("set seed as {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs(dataset_name=None, model_name=None):
    print("dataset:", dataset_name)
    data_config_path = "./config/data_config/" + dataset_name + ".yaml"
    model_config_path = "./config/model_config.yaml"
    trainer_config_path = "./config/trainer_config.yaml"

    with open(data_config_path, 'r') as data:
        data_config = yaml.load(data, Loader=yaml.FullLoader)
        data_config = data_config["base"]


    with open(model_config_path, 'r') as data:
        model_config = yaml.load(data, Loader=yaml.FullLoader)
        model_config = model_config[dataset_name + "-" + model_name]
    model_config["max_user_id"] = data_config["max_user_id"]
    model_config["max_item_id"] = data_config["max_item_id"]

    data_config["tau"] = model_config["tau"]

    with open(trainer_config_path, 'r') as data:
        trainer_config = yaml.load(data, Loader=yaml.FullLoader)
        trainer_config = trainer_config["base"]
    print(model_config)
    print(trainer_config)
    print(data_config)
    return {"data_config":data_config, "trainer_config":trainer_config, "model_config":model_config}


if __name__ == '__main__':
    seed_everything()
    set_logging()

    for wei in [2.0]:
        for temp in [0.1]:
            dataset_name = "ml-1m"
            model_name = "algcn"
            configs_dic = get_configs(dataset_name = dataset_name, model_name = model_name)
            data_config = configs_dic["data_config"]
            trainer_config = configs_dic["trainer_config"]
            model_config = configs_dic["model_config"]


            data_config["num_neg"] = 1
            trainer_config["gpu"] = 1
            data_config["tau"] = model_config["tau"]
            del model_config["tau"]


            model_config["n_layers"] = model_config["n_layers"]

            trainer = BaseTrainer(**trainer_config, model_config=model_config, data_config=data_config, uni_weight=wei, loss_temp=temp)
            trainer.train()











