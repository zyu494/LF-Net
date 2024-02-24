# LF-Net
## Lyft Dataset
To train your own model, you need to download the [Lyft Motion Prediction Dataset](https://level-5.global/download/). These files contain ```Training Dataset(8.4GB), validation Dataset (8.2GB), Aerial Map and Semantic Map```. You also need to download Lyft's [Python software kit](https://github.com/woven-planet/l5kit). The detailed tutorial can be found at https://woven-planet.github.io/l5kit/introduction.html. All data files should be stored in a single folder to match this structure: https://woven-planet.github.io/l5kit/dataset.html.
## Train Your Model
Run ```train.py``` to use single GPU to train the model.
```shell
python train.py 
```
Run ```multi_train.py``` to use multi GPU to train the model.
```shell
python -m torch.distributed.launch --nproc_per_node=2 multi_train.py 
```