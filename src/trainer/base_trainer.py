
from src.data_generator import *
import numpy as np
import torch
import torch.nn as nn
from src.metrics import evaluate_metrics
import src.models.loss_function as loss_function
from tqdm import tqdm
import torch.nn.functional as F

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
                 lossfn_temp=1.0,
                 uni_weight=0,
                 loss_temp=0.1,
                 save_emb_name=None,
                 **kwargs):
        print(kwargs)
        print("Building Base Trainer...")
        self.lossfn_temp = lossfn_temp
        self.cul_total_epoch = cul_total_epoch
        self.kwargs = kwargs
        self.weight_decay = weight_decay
        self.emb_lambda = emb_lambda
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer
        self._validation_metrics = metrics
        self.save_emb_name = save_emb_name

        #   build loss function:
        self.loss_fn = self.get_loss_fn(loss)
        self.infonce = loss_function.InfoNCE(loss_temp)
        #self.uniform_loss = loss_function.UniformLoss(1)
        #self.directAU = loss_function.DirectAU(gamma)
        self.uni_weight = uni_weight

        self.per_eval = per_eval
        self._best_metrics = 0
        self.lr = lr
        self.save_model_path = save_model_path
        self.is_pretrained = is_pretrained
        self.is_save_embedding = is_save_embedding
        self.device = self.set_device(gpu)

        #    build data:
        self.train_gen, self.valid_gen, test_gen = data_generator(**data_config)
        adj_mat = self.train_gen.adj_mat

        #    build model:
        model_config["device"] = self.device
        if model_config["model_name"]=="lightgcn" or model_config["model_name"]=="algcn":
            model_config["adj_mat"] = adj_mat
        self.model = self.build_model(model_config).to(self.device)


        self.optimizer = self.set_optimizer(self.model)


    def train(self):
        early_stop_patience = 0
        self.model.preprocess()

        for epoch in range(self.num_epochs):

            self.epoch = epoch
            print("************ Epoch={} start ************".format(epoch))

            #   training process:
            epoch_loss = 0
            with tqdm(total=len(self.train_gen)) as pbar:
                for batch_index, batch_data in enumerate(self.train_gen):
                    batch_data = [x.to(self.device) for x in batch_data]
                    epoch_loss += self._step(batch_data)
                    pbar.update(1)
            epoch_loss = epoch_loss / (len(self.train_gen))
            print("Train Loss: {:.6f}".format(epoch_loss))


            res_dic =  self.evaluate(self.model, self.train_gen, self.valid_gen)
            if self.check_stop(res_dic, epoch):
                break
        print("Training finished")
        print("best epoch: ",self._best_epoch)
        print(self._best_res)


    def _step(self, batch_data):
        model = self.model.train()
        user_id, pos_item_id, neg_item_id = batch_data[:3]

        self.optimizer.zero_grad()

        return_dict = model.forward(user_id, pos_item_id, neg_item_id)

        user_vec = return_dict["user_vec"]
        pos_item_vec = return_dict["pos_item_vec"]
        neg_item_vec = return_dict["neg_item_vec"]

        pos_y_pred = (pos_item_vec * user_vec).mean(dim=1)
        neg_y_pred = (neg_item_vec * user_vec).mean(dim=1)



        mf_loss = self.loss_fn(pos_y_pred, neg_y_pred)
        emb_loss = self.get_emb_loss(return_dict["embeds"])
        #uniform_loss = self.uniform_loss(user_vec.squeeze(1), pos_item_vec.squeeze(1))
        #directAU_loss = self.direct AU(user_vec.squeeze(1), pos_item_vec.squeeze(1))
        #loss = uniform_loss*self.uni_weight + emb_loss * self.emb_lambda + mf_loss
        if self.uni_weight>0:
            uniform_loss = self.infonce(user_vec.squeeze(1), pos_item_vec.squeeze(1))
            loss = mf_loss +  emb_loss * self.emb_lambda  +  uniform_loss*self.uni_weight
        else:
            loss = mf_loss + emb_loss * self.emb_lambda

        loss.backward()
        self.optimizer.step()

        return loss.item()



    def evaluate(self, model, train_generator, valid_generator, k=-1):
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

            if k==-1:
                valid_user2items_group = None
            else:
                valid_user2items_group = valid_generator.user2items_group_dict[k]
            val_logs = evaluate_metrics(train_generator.user2items_dict,
                                        valid_generator.user2items_dict,
                                        valid_generator.test_users,
                                        self._validation_metrics,
                                        user_embs=user_vecs,
                                        item_embs=item_vecs,
                                        valid_user2items_group=valid_user2items_group
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
            return loss_function.InfoNCELoss(temp=self.lossfn_temp)
        elif loss.lower()=="InfoNCE".lower():
            print("InfoNCE init.")
            return loss_function.InfoNCE()
        elif loss.lower()=="PairwiseLogisticLoss".lower():
            print("PairwiseLogisticLoss init.")
            return loss_function.PairwiseLogisticLoss()
        elif loss.lower()=="MarginalHingeLoss".lower():
            print("MarginalHingeLoss init.")
            return loss_function.MarginalHingeLoss()
        elif loss.lower()=="GumbelLoss".lower():
            print("GumbelLoss init.")
            return loss_function.GumbelLoss()

    def set_device(self, gpu=-1):
        if gpu>=0 and torch.cuda.is_available():
            device = torch.device("cuda:"+str(gpu))
        else:
            device = torch.device("cpu")
        logging.info(device)
        return device

    def set_optimizer(self, model):
        print("using: ",self.optimizer_name)
        params = []
        for m in model.modules():

            if isinstance(m, nn.Embedding):
                params.append({"params":m.parameters(),"lr":self.lr, "weight_decay":self.weight_decay})
            elif isinstance(m, nn.Linear):
                params.append({"params": m.parameters(), "lr": self.lr*0.1, "weight_decay":self.weight_decay})

        return getattr(torch.optim, self.optimizer_name)(model.parameters(), lr = self.lr)
        #return getattr(torch.optim, self.optimizer_name)(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        logging.info("saving weight successfully.")

    def set_scheduler(self):
        import torch.optim as optim
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, min_lr=self.lr*0.1)


    def build_model(self, model_config):
        from src.models.MF import MF
        from src.models.ALGCN import ALGCN

        model_dic = {
            "mf" : MF,
            "algcn" : ALGCN
        }
        return model_dic[model_config["model_name"]](**model_config)

    def save_embedding(self, model, path):
        torch.save(model.user_embedding.weight, path + "user_embedding.pt")
        torch.save(model.item_embedding.weight, path + "item_embedding.pt")
        logging.info("Saving embedding successfully.")

    def check_stop(self, res_dic, epoch):
        if res_dic["Recall(k=20)"] >= self._best_metrics:
            self._best_metrics = res_dic["Recall(k=20)"]
            self._best_res = res_dic
            self._best_epoch = epoch
            if self.save_emb_name!=None:
                self.model.save_gcn_embeds(self.save_emb_name)

            print("New best result!")
            self.early_stop_patience = 0
        else:
            self.early_stop_patience += 1

            if self.early_stop_patience >= 5:
                print("Early stopped!")
                return True
        return False



