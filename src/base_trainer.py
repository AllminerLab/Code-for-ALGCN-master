
from src.data_generator import *
import numpy as np
import torch
from src.metrics import evaluate_metrics
import src.models.loss_function as loss_function
from tqdm import tqdm

class BaseTrainer():
    def __init__(self,
                 optimizer="Adam",
                 gpu=-1,
                 metrics=None,
                 num_epochs=1000,
                 lr=None,
                 emb_lambda=None,
                 loss=None,
                 save_model_path=None,
                 is_pretrained=False,
                 is_save_embedding=False,
                 weight_decay=0,
                 cul_total_epoch=0,
                 per_eval=5,
                 model_config=None,
                 data_config=None,
                 **kwargs):
        print(kwargs)
        self.cul_total_epoch = cul_total_epoch
        self.kwargs = kwargs
        self.weight_decay = weight_decay
        self.emb_lambda = emb_lambda
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer
        self._validation_metrics = metrics
        self.loss_fn = self.get_loss_fn(loss)
        self.per_eval = per_eval
        self._best_metrics = 0
        self.lr = lr
        self.save_model_path = save_model_path
        self.is_pretrained = is_pretrained
        self.is_save_embedding = is_save_embedding
        self.device = self.set_device(gpu)

        #    build model:
        self.model = self.build_model(model_config).to(self.device)

        #    build data:
        self.train_gen, self.valid_gen, test_gen = data_generator(**data_config)

        self.optimizer = self.set_optimizer(self.model)


    def train(self):
        early_stop_patience = 0
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            logging.info("************ Epoch={} start ************".format(epoch))

            #   training process:
            epoch_loss = 0
            with tqdm(total=len(self.train_gen)) as pbar:
                for batch_index, batch_data in enumerate(self.train_gen):
                    batch_data = [x.to(self.device) for x in batch_data]
                    epoch_loss += self._step(batch_data)
                    pbar.update(1)
            epoch_loss = epoch_loss / (len(self.train_gen))
            logging.info("Train Loss: {:.6f}".format(epoch_loss))


            #   evaluate valid data:
            #if epoch%5 == 0:
            res_dic =  self.evaluate(self.model, self.train_gen, self.valid_gen)

            if res_dic["Recall(k=20)"]>=self._best_metrics:
                self._best_metrics = res_dic["Recall(k=20)"]
                logging.info("New best result!")
                if self.is_save_embedding:
                    self.save_embedding(self.model, self.save_model_path)
                early_stop_patience = 0
            else:
                early_stop_patience += 1
            if early_stop_patience >= 5:
                logging.info("Early stopped!")
                break
        logging.info("Training finished")


    def _step(self, batch_data):
        model = self.model.train()
        user_id, pos_item_id, neg_item_id = batch_data[:3]

        self.optimizer.zero_grad()
        neg_item_id = model.hard_negative_sample(user_id, neg_item_id)

        return_dict = model.forward(user_id, pos_item_id, neg_item_id)
        mf_loss = self.loss_fn(return_dict["pos_y_pred"], return_dict["neg_y_pred"])
        emb_loss = self.get_emb_loss(return_dict["embeds"])

        loss = mf_loss  +  emb_loss * self.emb_lambda
        loss.backward()
        self.optimizer.step()

        return loss.item()



    def evaluate(self, model, train_generator, valid_generator):
        logging.info("**** Start Evaluation ****")
        model.eval()
        with torch.no_grad():
            user_vecs = []
            item_vecs = []
            for user_batch in valid_generator.user_loader:
                user_vec = model.user_tower(user_batch.to(self.device))
                user_vecs.extend(user_vec.data.cpu().numpy())
            for item_batch in valid_generator.item_loader:
                item_vec = model.item_tower(item_batch.to(self.device))
                item_vecs.extend(item_vec.data.cpu().numpy())
            user_vecs = np.array(user_vecs, np.float64)
            item_vecs = np.array(item_vecs, np.float64)

            val_logs = evaluate_metrics(train_generator.user2items_dict,
                                        valid_generator.user2items_dict,
                                        valid_generator.test_users,
                                        self._validation_metrics,
                                        user_embs1=user_vecs,
                                        item_embs1=item_vecs,
                                        )
        return val_logs

    def get_emb_loss(self, embs):
        loss = 0
        for emb in embs:
            loss += torch.norm(emb) ** 2
        loss /= 2.0
        # batch size
        loss /= embs[0].shape[0]
        return loss


    def get_loss_fn(self, loss):
        if loss.lower()=="CosineContrastiveLoss".lower():
            print("CosineContrastiveLoss init.")
            return loss_function.CosineContrastiveLoss(self.kwargs.get("margin",0), self.kwargs.get("negative_weight"))
        elif loss.lower()=="InfoNCELoss".lower():
            print("InfoNCELoss init.")
            return loss_function.InfoNCELoss()
        elif loss.lower()=="InfoNCE".lower():
            print("InfoNCE init.")
            return loss_function.InfoNCE()
        elif loss.lower()=="PairwiseLogisticLoss".lower():
            print("PairwiseLogisticLoss init.")
            return loss_function.PairwiseLogisticLoss()
        elif loss.lower()=="MarginalHingeLoss".lower():
            print("MarginalHingeLoss init.")
            return loss_function.MarginalHingeLoss()

    def set_device(self, gpu=-1):
        if gpu>=0 and torch.cuda.is_available():
            device = torch.device("cuda:"+str(gpu))
        else:
            device = torch.device("cpu")
        logging.info(device)
        return device

    def set_optimizer(self, model):
        print("using: ",self.optimizer_name)
        return getattr(torch.optim, self.optimizer_name)(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        logging.info("saving weight successfully.")

    def set_scheduler(self):
        import torch.optim as optim
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, min_lr=self.lr*0.1)


    def build_model(self, model_config):
        from src.models.MF import MF
        from src.models.Lgcn import LightGCN

        model_dic = {
            "mf" : MF,
            "lightgcn" : LightGCN
        }
        return model_dic[model_config["model_name"]](**model_config)

    def save_embedding(self, model, path):
        torch.save(model.user_embedding.weight, path + "user_embedding.pt")
        torch.save(model.item_embedding.weight, path + "item_embedding.pt")
        logging.info("Saving embedding successfully.")




