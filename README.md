## Train MVSNet and MVSNet + novel up and down blocks in DEMVSNet on DTU which have been already added noise
### Train and test MVSNet, in **"./train.py"** and **"./eval.py"**, **args.model="mvsnet"**
* Train
```
args.model = "mvsnet"
args.trainpath = ``MVS_TRAINING``
args.trainlist = "./lists/dtu/train.txt"
args.testpath = ``MVS_TRAINING``
args.testlist = "./lists/dtu/test.txt"
args.batch_size = * # Set the appropriate value according to the GPU memory
args.resume = * # If set to True, you need to select the path of checkpoints
args.logdir = "./checkpoints" # set the path to save checkpoints，args.logdir 
``` 
* Test
```
args.model = "mvsnet"
args.testpath = "DTU_TESTING"
args.testlist = './lists/dtu/test.txt'
args.loadckpt = './checkpoints/***.ckpt' # Select the model to be tested
args.outdir = '*/outdir' # Set the path to save the results
args.batch_size = * # Set the appropriate value according to the GPU memory
``` 
### Train and test MVSNet + novel up and down blocks in DEMVSNet, in **"./train.py"** and **"./eval.py"**, **args.model="mvsnet"**
* You need to change **args.model** parameter to **mvsnet_denoise**
* The [MVSNet and MVSNet+Ours pretrained](https://drive.google.com/drive/folders/1Bh5xLb7tB_3-_jxCrPxnEp_aUMhklIDo?usp=sharing) on noisy images model, you can click the link to download.
