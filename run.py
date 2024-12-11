import yaml
import argparse
from modelR_fixed import GPTConfig
from transformers import AutoTokenizer  
import lightning as pl
import logging
from dataloader_parity import MyDataLoader
import os 
import platform
from lightning.pytorch.strategies import DDPStrategy
import re
from typing import Tuple
from model_light import MyModel
import psutil
import operator
import torch
vocab= {"pad":0, "a":1, "b":2, "eos": 3}
def _get_devices() :
    """
    Get the number of GPUs/CPUs for the Trainer to use.

    Returns
    -------
    Union[int, str]
        The number of GPUs/CPUs to use, or "auto" to let PyTorch Lightning
        determine the appropriate number of devices.
    """
    
    if any(
            operator.attrgetter(device + ".is_available")(torch)()
            for device in ["cuda", "backends.mps"]):
        return -1
    elif not (n_workers := n_workers()):
        return "auto"
    else:
        return n_workers
def n_workers() -> int:
    """
    Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process, scaled for the
    number of GPUs being used.

    On Windows and MacOS, we only use the main process. See:
    https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    https://github.com/pytorch/pytorch/issues/70344

    Returns
    -------
    int
        The number of workers.
    """
    # Windows or MacOS: no multiprocessing.
    if platform.system() in ["Windows", "Darwin"]:
        return 0
    # Linux: scale the number of workers by the number of GPUs (if present).
    try:
        n_cpu = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpu = os.cpu_count()
    return (
        n_cpu // n_gpu if (n_gpu := torch.cuda.device_count()) > 1 else n_cpu
    )


def train(model_config):
    logging.info('Training Mode Activated ')
    with open('training_args.yaml', 'r') as file:
        training_args = yaml.safe_load(file)
    dataModule = MyDataLoader(training_args["max_len"], 64, training_args["bz_per_gpu_train"], training_args["bz_per_gpu_test"], training_args["m"])
    train_dataloader=dataModule.train_dataloader()
    training_args["warmup_iters"] = int(len(train_dataloader)/(torch.cuda.device_count()*training_args["accumulate_grad_batches"])) *  training_args["warm_up_epochs"]
    training_args["max_iters"] = int(len(train_dataloader)/(torch.cuda.device_count()*training_args["accumulate_grad_batches"])) * int(training_args["max_epochs"])
    model = MyModel(model_args=model_config, training_args=training_args)
    
    if training_args["save_model"]:
        callbacks = [
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=training_args["model_save_folder_path"],
                save_top_k=-1,
                save_weights_only=False,
                every_n_train_steps=training_args["every_n_train_steps"],
            )
        ]
    else:
        callbacks = []
    import time
    if training_args["enable_neptune"]:
        callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch'))
        neptune_logger = pl.pytorch.loggers.NeptuneLogger(
            project=training_args["neptune_project"],
            api_token=training_args["neptune_api_token"],
            log_model_checkpoints=False,
            custom_run_id= str(time.time()),
            name= str(time.time()),
        )
    trainer = pl.Trainer(
        
        # reload_dataloaders_every_n_epochs=1,
        enable_model_summary= True,
        accelerator="auto",
        #devices="auto",
        callbacks=callbacks,
        devices=_get_devices(),
        num_nodes=training_args["n_nodes"],
        logger=neptune_logger if training_args["enable_neptune"] else None,
        max_epochs=training_args["max_epochs"],
        #num_sanity_val_steps=training_args["num_sanity_val_steps"],
        strategy= "ddp", #DDPStrategy(static_graph=True),
        gradient_clip_val=training_args["gradient_clip_val"],
        gradient_clip_algorithm=training_args["gradient_clip_algorithm"],
        accumulate_grad_batches=training_args["accumulate_grad_batches"],
        sync_batchnorm=training_args["sync_batchnorm"],
        num_sanity_val_steps=0
    )
    
    trainer.fit(model, 
                datamodule=dataModule)
        
    
    
    
    

    
    
    
def predict_one(model_args):
    with open('training_args.yaml', 'r') as file:
        training_args = yaml.safe_load(file)
    model = MyModel().load_from_checkpoint(training_args["model_filename"]).to("cuda")
        
    input_text = input("enter your strings for parity check: ")
    input_ids = [[vocab[each] for each in input_text] + [vocab["eos"]]]
    attention_mask = [[1 for _ in range(len(input_ids))]]
    input_ids = torch.tensor(input_ids).to("cuda")
    attention_mask = torch.tensor(attention_mask).to("cuda")
    
    
    output  = model.predict_one(input_ids, attention_mask)
    print("prediction probaility for each class is: ", output)
    
    
        
    
    
    
    
def predict():
    pass
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='running mode: train/validate/predict')
    
    args = parser.parse_args()
    
    #tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    # with open('model_config.yaml', 'r') as file:
    #     model_arg = yaml.safe_load(file)
    
    #model_arg["vocab_size"] = tokenizer.vocab_size
    with open('training_args.yaml', 'r') as file:
        training_args = yaml.safe_load(file)
    model_config = GPTConfig(block_size=training_args["block_size"], vocab_size=training_args["vocab_size"], n_layer=training_args["n_layer"], n_head=training_args["n_head"],
                             n_embd=training_args["n_embd"])
    
    
    
    if not args.mode:
        raise RuntimeError("mode not provided")
        
    if args.mode == "train":
        train(model_config)    
    if args.mode == "denovo":
        predict_one(model_config)
main()
