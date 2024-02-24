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

def main():
    
    cfg = load_config_data("./config/lyft.yaml")
    num_workers = len(os.sched_getaffinity(0)) // torch.cuda.device_count()
    dm = LocalDataManager(None)
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_zarr = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()

    semantic_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
    dataset_meta_key = dm.require(cfg["raster_params"]["dataset_meta_key"])
    dataset_meta = load_metadata(dm.require(dataset_meta_key))
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    map_api = MapAPI(dm.require(semantic_map_path), world_to_ecef)
    meta_manager = LyftManager(cfg, dm)
    
    train_dataset = LyftDataset(cfg, 'train', meta_manager, train_zarr,map_api)
    val_dataset = LyftDataset_val(cfg, 'val', meta_manager, val_zarr,map_api)

    epoch=1000
    train_cfg = cfg['train_data_loader']
    val_cfg = cfg['val_data_loader']
    model= Model(cfg['model_params'])
    train_loader = DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=32,
                                        num_workers=16,
                                        #prefetch_factor=2,
                                        pin_memory=True,
                                        drop_last=True
                                        )

    val_loader = DataLoader(val_dataset,
                                        shuffle=val_cfg["shuffle"],
                                        batch_size=32,
                                        num_workers=16,
                                        drop_last=True
                                        )

    optim = AdamW(model.parameters(), lr=0.005, betas=(0.9, 0.999), weight_decay=0.01)
    optm_schedule = ScheduledOptim(
            optim,
            0.005,#lr
            n_warmup_epoch=20, #warmup_epoch,
            update_rate=5, #lr_update_freq,
            decay_rate=0.3 #lr_decay_rate
        )

    best_train_loss=None

    for i in range(epoch):
        train_losses=[]
        #model=torch.load('current_best_model')
        model=model.to(device)
        model.train()
        for j, data in tqdm(enumerate(train_loader)):
            data={key:data[key].to(device) for key in data}
            optm_schedule.zero_grad()
            loss = compute_loss(model,data)
            loss.backward()
            optim.step()
            loss=loss.cpu()
            train_losses.append(loss.item())
        train_loss = sum(train_losses)/len(train_losses)
    
        if best_train_loss is None or best_train_loss > train_loss:
            best_train_loss = train_loss
            # save the best model
            with open('pre_model/'+str(i+1)+'_epoch_'+'best_model', 'wb') as f:
                torch.save(model, f)
        #train_loss_his.append(train_loss)
        print("Epoch: {0}| Train Loss: {1:.7f}".format(i + 1, train_loss))
        print(min(train_losses),max(train_losses))

        if i%5==0:
            test_losses=[]
            pos_loss=[]
            model=torch.load('./pre_model/'+str(i+1)+'_epoch_best_model')
            model=model.to(device)
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for j,data in tqdm(enumerate( val_loader)):
                    off_tensor=torch.tensor([[0.0,0.0,0.0]]).to(device)
                    data={key:data[key].to(device) for key in data}
                    pred_dict = model(data)
                    gt = {
                        "target_prob": data["candidate_gt"].squeeze(2),
                        "offset": data["offset"]
                    }
                    loss,loss_dict = Loss(pred_dict, gt)
                    test_losses.append(loss.cpu().item())
                    for i in range(32):
                        try: 
                            id=data["candidate_gt"][i].nonzero()[0][0]    
                            pos_gt=data["target_candidates"][i][id]+data["offset"][i]
                            sample_id=torch.argmax(pred_dict["target_prob"][i])
                            pos=data["target_candidates"][i][sample_id]
                        except Exception:
                            pos_gt=torch.tensor([0.0,0.0,0.0]).to(device)
                            pos=torch.tensor([0.0,0.0,0.0]).to(device)
                        off=pos-pos_gt
                        off_tensor=torch.cat((off_tensor, off.unsqueeze(0)), 0) 
                    off_batch=off_tensor.mean(dim=0)
                    pos_loss.append(off_batch)
                    
                test_loss=sum(test_losses)/len(test_losses)
                tar_loss=sum(pos_loss)/len(pos_loss)
                #print('loss|%.4f, pred|%.5f'%(test_loss, 100. * correct / len(val_loader.dataset)))
                print(test_loss,tar_loss)
    
def compute_loss(model, data):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n = len(data["target_candidates"])
        pred_dict = model(data)
    
        gt = {
            "target_prob": data["candidate_gt"].squeeze(2),
            "offset": data["offset"]
        }

        return Loss(pred_dict, gt)

if __name__=='__main__':
    main()