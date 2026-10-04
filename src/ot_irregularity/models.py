import torch
from torch import nn
import time,sys


def resolve_device(requested="auto"):
    requested=str(requested).lower()
    if requested=="auto":return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device=torch.device(requested)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable in this PyTorch environment")
    return device

class Autoencoder(nn.Module):
    def __init__(self,n_features,latent_dim=16):
        super().__init__();h1=max(8,min(128,n_features*4));h2=max(4,min(64,n_features*2));latent=max(2,min(latent_dim,h2))
        self.net=nn.Sequential(nn.Linear(n_features,h1),nn.ReLU(),nn.Dropout(.05),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,latent),nn.ReLU(),nn.Linear(latent,h2),nn.ReLU(),nn.Linear(h2,h1),nn.ReLU(),nn.Linear(h1,n_features))
    def forward(self,x):return self.net(x)

def train_ae(x,validation_x,cfg,seed,path,progress_path=None):
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True,warn_only=True)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device=resolve_device(cfg.get("device","auto"));model=Autoencoder(x.shape[1],cfg.get("latent_dim",16)).to(device);opt=torch.optim.Adam(model.parameters(),lr=cfg.get("learning_rate",.001),weight_decay=1e-5);tensor=torch.tensor(x,dtype=torch.float32,device=device);validation=torch.tensor(validation_x,dtype=torch.float32,device=device);best=float("inf");best_epoch=0;best_state=None;patience=int(cfg.get("patience",10));stale=0;epochs=int(cfg.get("epochs",100));steps_requested=int(cfg["steps"]) if cfg.get("steps") is not None else None;steps_completed=0;started=time.perf_counter();path=__import__("pathlib").Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(tensor),device=device)
        for ids in order.split(int(cfg.get("batch_size",128))):
            batch=tensor[ids];opt.zero_grad();loss=((model(batch)-batch)**2).mean();loss.backward();opt.step();steps_completed+=1
            if steps_requested is not None and steps_completed>=steps_requested:break
        model.eval()
        with torch.no_grad():score=((model(validation)-validation)**2).mean().item()
        if score<best:
            best=score;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        log_interval=int(cfg.get("log_interval",max(1,epochs//100)))
        if log_interval>0 and steps_completed%log_interval==0:
            elapsed=time.perf_counter()-started
            gpu={"memory_allocated_mb":torch.cuda.memory_allocated(device)/1048576,"memory_reserved_mb":torch.cuda.memory_reserved(device)/1048576,"memory_total_mb":torch.cuda.get_device_properties(device).total_memory/1048576} if device.type=="cuda" else {}
            event={"phase":"autoencoder_training","timestamp":time.time(),"steps_requested":steps_requested,"steps_completed":steps_completed,"epochs_requested":epochs,"epochs_completed":epoch,"best_epoch":best_epoch,"best_validation_loss":best,"device":str(device),"elapsed_seconds":elapsed,**gpu}
            print(f"autoencoder step {steps_completed}/{steps_requested or 'uncapped'}; epoch={epoch}/{epochs}; best_epoch={best_epoch}; validation_loss={best:.8g}; device={device}; elapsed={elapsed:.1f}s",file=sys.stderr,flush=True)
            if progress_path is not None:
                with __import__("pathlib").Path(progress_path).open("a",encoding="utf-8") as stream:stream.write(__import__("json").dumps(event)+"\n")
        if steps_requested is not None and steps_completed>=steps_requested:break
        if stale>=patience:break
    model.load_state_dict(best_state);model.eval();model.training_metadata={"device":str(device),"epochs_requested":epochs,"epochs_completed":epoch,"steps_requested":steps_requested,"steps_completed":steps_completed,"best_epoch":best_epoch,"best_validation_loss":best,"training_seconds":time.perf_counter()-started}
    if progress_path is not None:
        event={"phase":"autoencoder_completed","timestamp":time.time(),**model.training_metadata}
        with __import__("pathlib").Path(progress_path).open("a",encoding="utf-8") as stream:stream.write(__import__("json").dumps(event)+"\n")
    torch.save({"state_dict":best_state,"n_features":x.shape[1],"latent_dim":cfg.get("latent_dim",16),"best_validation_loss":best,"best_epoch":best_epoch,"epochs_requested":epochs,"epochs_completed":epoch,"steps_requested":steps_requested,"steps_completed":steps_completed,"device":str(device)},path)
    return model

def load_ae(path,device="cpu"):
    device=resolve_device(device);c=torch.load(path,map_location="cpu",weights_only=True);m=Autoencoder(c["n_features"],c["latent_dim"]);m.load_state_dict(c["state_dict"]);m.to(device);m.eval();return m

def ae_errors(model,x):
    device=next(model.parameters()).device
    with torch.no_grad():return (model(torch.tensor(x,dtype=torch.float32,device=device)).cpu().numpy()-x)**2
