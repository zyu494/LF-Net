import sys
import os
import torch
import numpy as np
os.environ["L5KIT_DATA_FOLDER"] = "The path of your data folder"  
from l5kit.data import LocalDataManager
from l5kit.configs import load_config_data
from l5kit.configs.config import load_metadata
from l5kit.data.map_api import MapAPI
from vectorize.Lyft_load import ChunkedDataset
from vectorize.Lyft_dataset import LyftDataset
from vectorize.Lyft_dataset_val import LyftDataset_val
from vectorize.Lyft_manager import LyftManager
from optim_schedule import ScheduledOptim
from model.model import Model
from torch.utils.data import DataLoader
#from torch_geometric.data import DataLoader
from torch.optim import Adam, AdamW
from util.loss import Loss
from tqdm import tqdm
import os
import torch.distributed as dist
import argparse
from torch.utils.data.distributed import DistributedSampler
import tempfile


def main(args):
    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    weights_path = args.weights
    batch_size = args.batch_size
    args.lr *= args.world_size  
    checkpoint_path = ""


    cfg = load_config_data("./config/lyft.yaml")
    num_workers = len(os.sched_getaffinity(0)) // torch.cuda.device_count()
    dm = LocalDataManager(None)
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    
    #device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_zarr = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()

    semantic_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
    dataset_meta_key = dm.require(cfg["raster_params"]["dataset_meta_key"])
    dataset_meta = load_metadata(dm.require(dataset_meta_key))
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    map_api = MapAPI(dm.require(semantic_map_path), world_to_ecef)
    meta_manager = LyftManager(cfg, dm)
    
    train_dataset = LyftDataset(cfg, 'val', meta_manager, train_zarr,map_api)
    val_dataset = LyftDataset_val(cfg, 'val', meta_manager, val_zarr,map_api)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
    
    epoch=1000
    train_cfg = cfg['train_data_loader']
    val_cfg = cfg['val_data_loader']
    model= Model(cfg['model_params']).to(device)
    train_loader = DataLoader(train_dataset,
                                        #shuffle=train_cfg["shuffle"],
                                        #batch_size=train_cfg["batch_size"] // torch.cuda.device_count(),
                                        batch_sampler=train_batch_sampler,
                                        num_workers=16,
                                        #prefetch_factor=2,
                                        pin_memory=True
                                        #drop_last=True
                                        )

    val_loader = DataLoader(val_dataset,
                                        shuffle=val_cfg["shuffle"],
                                        batch_size=32,
                                        num_workers=16,
                                        sampler=val_sampler,
                                        #prefetch_factor=2,
                                        #pin_memory=True,
                                        drop_last=True
                                        )

    if os.path.exists(weights_path):
        print('load weight!')
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        if args.syncBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
    
    optim = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    optm_schedule = ScheduledOptim(
            optim,
            args.lr,#lr
            n_warmup_epoch=20, #warmup_epoch,
            update_rate=5, #lr_update_freq,
            decay_rate=0.3 #lr_decay_rate
        )

    best_train_loss=None
   
    for i in range(epoch):
        train_sampler.set_epoch(i)

        train_losses=[]
        #model=torch.load('current_best_model')
        model.train()
        for j, data in tqdm(enumerate(train_loader)):
            data={key:data[key].to(device) for key in data}
            optm_schedule.zero_grad()
            loss = compute_loss(model,data)
            loss.backward()
            loss = reduce_value(loss, average=True)
            optim.step()
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)
            train_losses.append(loss.cpu().item())
        # model.eval()
        # with torch.no_grad():
        #      for m,eval_data in tqdm(enumerate(val_loader)):
        #           preds=model(eval_data)
        train_loss = sum(train_losses)/len(train_losses)
    
        if best_train_loss is None or best_train_loss > train_loss:
            best_train_loss = train_loss
            # save the best model
            torch.save(model.module.state_dict(), "pre_state_dict/model-{}.pth".format(i+1))
            with open('try_model/'+str(i+1)+'_epoch_'+'best_model', 'wb') as f:
                torch.save(model.module, f)

        #train_loss_his.append(train_loss)
        if is_main_process():
            print("Epoch: {0}| Train Loss: {1:.7f}".format(i + 1, train_loss))
            print(min(train_losses),max(train_losses))

    
def compute_loss(model, data):
        
        n = len(data["target_candidates"])
        pred_dict = model(data)
    
        gt = {
            "target_prob": data["candidate_gt"].squeeze(2),
            "offset": data["offset"]
        }

        return Loss(pred_dict, gt) ##########

def init_distributed_mode(args):
    if'RANK'in os.environ and'WORLD_SIZE'in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif'SLURM_PROCID'in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu) 
    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend,init_method=args.dist_url,world_size=args.world_size,rank=args.rank)
    dist.barrier()  

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
    
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():

    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('--syncBN', type=bool, default=True)

    parser.add_argument('--data-path', type=str, default="/home/wz/data_set/processed")

    parser.add_argument('--freeze-layers', type=bool, default=False)

    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
  
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--weights', type=str, default='pre_state_dict/model-0.pth',
                        help='initial weights path')
    opt = parser.parse_args()

    main(opt)