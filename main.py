import argparse
import yaml
import os
import torch
from loguru import logger
from torch import optim
import wandb
import time
import numpy as np
from torchvision.transforms import v2

from mae import MAE
from dataloader import (
    get_dataloaders,
    imagenet_mean,
    imagenet_std,
)
from utils import (
    separate_weight_decay_weights,
    lrScheduler,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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
        default="trained_models/mae_checkpoint.pt",
    )
    parser.add_argument(
        "--load_from_ckpt",
        type=str,
    )
    parser.add_argument(
        "--is_distributed",
        action="store_true",
        default=False,
        help="If the training is multi gpu"
    )

    return parser.parse_args()

def train_one_iter(mae_model, batch, device, num_mini_batch, scaler, gpu_transforms, is_distributed, is_accumulating):
    xb = batch[0].to(device, non_blocking=True)
    xb = gpu_transforms(xb)
    if is_distributed and is_accumulating:
        with mae_model.no_sync():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                batch_loss, _ = mae_model.forward(xb)
            mini_batch_loss = batch_loss / num_mini_batch
            scaler.scale(mini_batch_loss).backward()
    else:
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
        loss_iters,
        gpu_transforms,
        is_distributed,
    ):
    val_loss = 0.0
    curr_epoch = 0

    if is_distributed:
        val_dataloader.sampler.set_epoch(curr_epoch)
    val_dataloader_iter = iter(val_dataloader)

    mae_model.eval()
    for j in range(loss_iters):
        try:
            batch = next(val_dataloader_iter)
        except StopIteration:
            #logger.info(f'val dataloader exhaused, resetting')
            curr_epoch += 1
            if is_distributed:
                val_dataloader.sampler.set_epoch(curr_epoch)
            val_dataloader_iter = iter(val_dataloader)
            batch = next(val_dataloader_iter)

        xb = batch[0].to(device, non_blocking=True)
        xb = gpu_transforms(xb)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            batch_loss, _ = mae_model.forward(xb)
        val_loss += batch_loss.detach().item()/loss_iters
    mae_model.train()
    return val_loss


def main(args):
    config_file = "configs.yaml"
    _workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())

    with open(os.path.join(_workspace, config_file), 'r') as file:
        config = yaml.safe_load(file)

    logger.info(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.is_distributed:
        dist.init_process_group("nccl")       
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = int(os.environ["WORLD_SIZE"])
        logger.info(f"rank {dist.get_rank()}: intialization done")
    else:
        world_size = 1
        local_rank = 0


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
    if args.is_distributed:
        training_model = DDP(mae_model.to(local_rank), device_ids=[local_rank])
        logger.info(f"rank {dist.get_rank()}: model init done")
    else:
        training_model = mae_model.to(device)

    if args.load_from_ckpt:
        model_load_path = os.path.join(_workspace, args.load_from_ckpt)
        checkpoint = torch.load(model_load_path)
        training_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    train_dataloader, val_dataloader = get_dataloaders(
                                        config["dataset_dir"], 
                                        config["train_folder"], 
                                        config["val_folder"],
                                        config["batch_size"],
                                        args.is_distributed,
    )
    
    if args.is_distributed:
        logger.info(f"rank {dist.get_rank()}: dataloaders done")

    if config['max_iter'] % world_size != 0:
        logger.warning(f'The number of total iterations are not divisible by world size, training iters will be smaller than the expected number')
    
    if config['num_mini_batch'] // world_size == 0 or config['num_mini_batch'] % world_size != 0:
        logger.info(f'Mini batch size needs to be divisible by total number of gpus')
        return

    max_iter = config['max_iter'] // world_size
    num_mini_batch = config['num_mini_batch'] // world_size

    #TODO: change peake warmup iter
    peak_warmup_iter = max_iter//20
    
    if not args.is_distributed or (args.is_distributed and dist.get_rank() == 0):
        run = wandb.init(
            entity="models_deepanshi",
            project="w8_mae_benchmarking",
            id=args.wandb_id,
            resume="allow",
            config={
                "learning_rate": config["max_lr"],
                "architecture": "MAE",
                "epochs": config["max_iter"],
                "batch_size": config["batch_size"],
                "warmup_steps": peak_warmup_iter//num_mini_batch,
            },
        )
        run.define_metric("opt_step")
        run.define_metric("*", step_metric="opt_step")

    #separte out weight decay params
    weight_decay_params, non_weight_decay_params = separate_weight_decay_weights(training_model)
    
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
                        int(peak_warmup_iter//num_mini_batch), 
                        float(config['min_lr']), 
                        int(max_iter// num_mini_batch)
    )

    if args.load_from_ckpt:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iter'] + 1
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        curr_epoch = checkpoint['epoch']

    else:
        start_iter = 0
        best_val_loss = torch.inf
        curr_epoch = 0

    if args.is_distributed:
        train_dataloader.sampler.set_epoch(curr_epoch)
    train_dataloader_iter = iter(train_dataloader)

    batch_size = config["batch_size"]

    mini_batch_running_loss = 0.0
    training_model.train()
    
    gpu_transforms = v2.Compose([
        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(imagenet_mean, imagenet_std),
    ])

    all_step_img_per_sec = []
    mini_batch_train_itr_time = 0
    if args.is_distributed:
        current_device_rank = dist.get_rank()
    else:
        current_device_rank = local_rank

    torch.cuda.synchronize(current_device_rank)

    for i in range(start_iter, max_iter):
        if (i)%num_mini_batch == 0:
            torch.cuda.synchronize(current_device_rank)
            temp_start_time = time.time()

        ##train one iter
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            #logger.info(f'train dataloader exhaused, resetting')
            curr_epoch += 1
            if args.is_distributed:
                train_dataloader.sampler.set_epoch(curr_epoch)
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)

        ## start iteration time
        
        batch_loss = train_one_iter(
            training_model,
            batch,
            device,
            num_mini_batch,
            scaler,
            gpu_transforms,
            args.is_distributed,
            False if (i+1)%num_mini_batch == 0 else True,
        )
        ## finish iteration time

        mini_batch_running_loss += batch_loss.detach()
        opt_step =(i + 1) // num_mini_batch
        if (i+1)%num_mini_batch == 0:
            scaler.step(optimizer)     
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            current_lrs = lr_scheduler.get_lr() if lr_scheduler else [min_lr, min_lr]

            if args.is_distributed:
                global_loss = mini_batch_running_loss.clone()
                dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)

                torch.cuda.synchronize(current_device_rank)
                temp_end_time = time.time()

                mini_batch_train_itr_time += (temp_end_time - temp_start_time)
                mini_batch_train_itr_time /= num_mini_batch
                global_mini_batch_train_itr_time = torch.tensor(mini_batch_train_itr_time).to(dist.get_rank())
                dist.reduce(global_mini_batch_train_itr_time, dst=0, op=dist.ReduceOp.AVG)
                
                if dist.get_rank() == 0:
                    images_per_sec_metric = (world_size*batch_size)/global_mini_batch_train_itr_time.detach().item()
                    all_step_img_per_sec.append(images_per_sec_metric)
                    run.log(
                        {
                            "train_loss": global_loss.item(),
                            "lr_weighted_decay": current_lrs[0],
                            "train_time_per_mini_batch": global_mini_batch_train_itr_time.detach().item(), 
                            "opt_step": opt_step,
                            "images_per_sec": images_per_sec_metric,
                        }
                    )
            else:
                torch.cuda.synchronize(current_device_rank)
                temp_end_time = time.time()

                mini_batch_train_itr_time += (temp_end_time - temp_start_time)
                mini_batch_train_itr_time /= num_mini_batch

                run.log(
                    {
                        "train_loss": mini_batch_running_loss.item(),
                        "lr_weighted_decay": current_lrs[0],
                        "train_time_per_mini_batch": mini_batch_train_itr_time, 
                        "opt_step": opt_step,
                        "images_per_sec": batch_size/mini_batch_train_itr_time,
                    }
                )
            mini_batch_running_loss = 0.0
            mini_batch_train_itr_time = 0.0
            temp_start_time = 0.0
            temp_end_time = 0.0

            
        ##add val step code
        if (i)%config['val_loss_compute_step'] == 0:
            val_loss = get_val_loss(
                    training_model,
                    val_dataloader, 
                    device,
                    config['loss_iters'],
                    gpu_transforms,
                    args.is_distributed,
            )
            val_loss = torch.tensor(val_loss).to(device)
            if args.is_distributed:
                global_val_loss = val_loss.clone()
                dist.all_reduce(global_val_loss, op=dist.ReduceOp.AVG)
                if dist.get_rank() == 0:
                    logger.info(global_val_loss)
                    run.log(
                        {
                            "val_loss": global_val_loss.item(),
                            "opt_step": opt_step,
                        }
                    )
                val_loss = global_val_loss
            else:
                logger.info(val_loss)
                run.log(
                    {
                        "val_loss": val_loss.item(),
                        "opt_step": opt_step,
                    }
                )
                            

            if val_loss < best_val_loss:
                # TODO: checkpointing should be done async with torch
                """if not args.is_distributed or (args.is_distributed and dist.get_rank() == 0):
                    checkpoint_dict = {
                        'model_state_dict': training_model.state_dict(),
                        'epoch': curr_epoch,
                        'iter': i,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'best_val_loss': val_loss.item(),
                        'scaler_state_dict': scaler.state_dict(),
                        'world_size': world_size,
                    }
                    
                    torch.save(checkpoint_dict, os.path.join(_workspace, args.model_save_path))
                """
                best_val_loss = val_loss

    if dist.get_rank() == 0:
        median_img_per_sec = np.median(np.array(all_step_img_per_sec))
        std_img_per_sec = np.std(np.array(all_step_img_per_sec))

        logger.info(["mean image per second", mean_img_per_sec])
        logger.info(["std image per second", std_img_per_sec])


    if args.is_distributed:
        dist.destroy_process_group()

        

if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
