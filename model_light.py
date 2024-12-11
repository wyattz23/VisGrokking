from modelR_fixed import GPT
import torch
import lightning as pl
#from torcheval.metrics.text import Perplexity
import numpy as np
class MyModel(pl.LightningModule):
    def __init__(self, model_args, training_args ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT(model_args)
        self.model_args = model_args
        self.training_args = training_args
        self.lm_head = torch.nn.Linear(model_args.n_embd, self.training_args["m"], bias=False)
        self.celoss = torch.nn.CrossEntropyLoss()
        #self.perp = Perplexity(-100) #might check device
    def predict_one(self, input_ids, mask):
        output = self.model(idx= input_ids, padding_mask=mask)
        pred= self.lm_head(output[0, -1, :])
        
        
        return pred.tolist()
    def training_step(self, batch, *args, mode="train"):
        input_ids, labels, attention_mask = batch
        outputs = self.model(
            idx = input_ids,
            padding_mask = attention_mask
        )
        indices = attention_mask.sum(dim=1) - 1
        hidden_states = outputs[torch.arange(outputs.size(0)),  indices, :] #bz, hidden
        
        pred= self.lm_head(hidden_states) #  bz, m
        predicted_classes = torch.argmax(pred, dim=1)
        correct_predictions = (predicted_classes == labels).sum().item()
        batch_size = labels.size(0)
        
        accuracy = correct_predictions / batch_size
        
        #pred = logits.view(-1, self.model_args.vocab_size) #
        #labels = labels.view(-1)
        
        
        loss = self.celoss(pred, labels)
        if (mode == "train"):
            
            if(not self.logger==None):
                #print("logging train")
                #self.logger.experiment["train_perp"].append(perp)
                self.logger.experiment["train_CELoss_step"].append(loss.detach())
            self.log(
                "train/CELoss",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False
            )
            self.log(
                "train/Acc",
                accuracy,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False
            )
        if (mode != "train"):
            self.log(
                "Test/Acc",
                accuracy,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False
            )
            
        return loss
    def validation_step(self, batch, batch_idx=None, dataloader_idx=None):
        loss = self.training_step(batch, mode="valid")
        self.log(
            "valid/CELoss" ,
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False
        )
        return loss
    
    def configure_optimizers(
        self, ) :
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_args["learning_rate"] )
        
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.training_args["warmup_iters"],
                                             max_iters=self.training_args["max_iters"]) 
        #return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}
        return optimizer

        
        

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int,
                 max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):

        # Cosine annealing after a constant period
        # Author: Sheng Xu
        # Date: 20230214

        decay=self.warmup/self.max_iters
        if epoch <= self.warmup:
            lr_factor = 1
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * (
                (epoch - (decay * self.max_iters)) / ((1-decay) * self.max_iters))))


        return lr_factor
