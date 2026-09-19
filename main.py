import argparse
import yaml
import os
import torch
from loguru import logger
from torch import optim
import wandb

from experimental.deepanshi.mae_reproduction.mae import MAE
from experimental.deepanshi.mae_reproduction.dataloader import get_dataloaders
from experimental.deepanshi.mae_reproduction.utils import (
    separate_weight_decay_weights,
    lrScheduler,
)

def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable_warmup",
        action="store_true", 
        help="Disable the warmup phase before running"
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="experimental/deepanshi/mae_reproduction/trained_models/mae_checkpoint.pt",
    )
    parser.add_argument(
        "--load_from_ckpt",
        type=str,
    )

    return parser.parse_args()

def train_one_iter(mae_model, batch, device, num_mini_batch, scaler):
    xb = batch[0].to(device, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        batch_loss, _ = mae_model.forward(xb)
    mini_batch_loss = batch_loss / num_mini_batch
    scaler.scale(mini_batch_loss).backward()
    return mini_batch_loss

@torch.no_grad()
def get_val_loss(
        mae_model, 
        val_dataloader,
        device, 
        loss_iters
    ):
    val_loss = 0.0
    val_dataloader_iter = iter(val_dataloader)

    mae_model.eval()
    for j in range(loss_iters):
        try:
            batch = next(val_dataloader_iter)
        except StopIteration:
            #logger.info(f'val dataloader exhaused, resetting')
            val_dataloader_iter = iter(val_dataloader)
            batch = next(val_dataloader_iter)

        xb = batch[0].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            batch_loss, _ = mae_model.forward(xb)
        val_loss += batch_loss.detach().item()/loss_iters
    mae_model.train()
    return val_loss


def main(args):
    config_file = "experimental/deepanshi/mae_reproduction/configs.yaml"
    _workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())

    with open(os.path.join(_workspace, config_file), 'r') as file:
        config = yaml.safe_load(file)

    logger.info(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mae_model = MAE(encoder_num_blocks=config['encoder_num_blocks'],
        decoder_num_blocks=config['decoder_num_blocks'],
        encoder_num_heads=config['encoder_num_heads'],
        decoder_num_heads=config['decoder_num_heads'],
        num_emb_encoder=config['num_emb_encoder'],
        num_emb_decoder=config['num_emb_decoder'],
        head_size=config['head_size'],
        patch_size=config['patch_size'],
        input_size=config['input_size'],
        device=device,
    )
    mae_model.to(device)

    if args.load_from_ckpt:
        model_load_path = os.path.join(_workspace, args.load_from_ckpt)
        checkpoint = torch.load(model_load_path)
        mae_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    train_dataloader, val_dataloader = get_dataloaders(
                                        config["dataset_dir"], 
                                        config["train_folder"], 
                                        config["val_folder"],
                                        config["batch_size"],
    )

    #TODO: change peake warmup iter
    peak_warmup_iter = config['max_iter']//20
    
    run = wandb.init(
        entity="models_deepanshi",
        project="mae_implementation",
        id=args.wandb_id,
        resume="allow",
        config={
            "learning_rate": config["max_lr"],
            "architecture": "MAE",
            "epochs": config["max_iter"],
            "batch_size": config["batch_size"],
            "warmup_steps": peak_warmup_iter//config['num_mini_batch'],
        },
    )

    #separte out weight decay params
    weight_decay_params, non_weight_decay_params = separate_weight_decay_weights(mae_model)
    
    optimizer = optim.AdamW([
        {
            'params': weight_decay_params,
            'weight_decay': config['weight_decay'],
            'lr': config['max_lr']
        },
        {
            'params': non_weight_decay_params,
            'weight_decay': 0.0,
            'lr': config['max_lr']
        }
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    #lr scheduler
    lr_scheduler = lrScheduler(
                        optimizer, 
                        int(peak_warmup_iter//config['num_mini_batch']), 
                        float(config['min_lr']), 
                        int(config['max_iter']// config['num_mini_batch'])
    )

    if args.load_from_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    else:
        start_epoch = 0
        best_val_loss = torch.inf

    train_dataloader_iter = iter(train_dataloader)

    mini_batch_running_loss = 0.0
    mae_model.train()
    
    for i in range(start_epoch, config['max_iter']):
        
        ##train one iter
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            #logger.info(f'train dataloader exhaused, resetting')
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)

        batch_loss = train_one_iter(
            mae_model,
            batch,
            device,
            config['num_mini_batch'],
            scaler,
        )
        mini_batch_running_loss += batch_loss.detach().item()
        if (i+1)%config['num_mini_batch'] == 0:
            ##grad clipping ?
            scaler.step(optimizer)     
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            current_lrs = lr_scheduler.get_lr() if lr_scheduler else [min_lr, min_lr]
            run.log(
                {
                    "train_loss": mini_batch_running_loss,
                    "lr_weighted_decay": current_lrs[0],
                }
            )
            mini_batch_running_loss = 0.0

            
        ##add val step code
        if (i)%config['val_loss_compute_step'] == 0:
            val_loss = get_val_loss(
                    mae_model,
                    val_dataloader, 
                    device,
                    config['loss_iters'],
            )
            logger.info(val_loss)
            run.log(
                {
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_val_loss:
                checkpoint_dict = {
                    'model_state_dict': mae_model.state_dict(),
                    'epoch': i,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': val_loss,
                    'scaler_state_dict': scaler.state_dict(),
                }
                torch.save(checkpoint_dict, os.path.join(_workspace, args.model_save_path))
                best_val_loss = val_loss

        

if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)