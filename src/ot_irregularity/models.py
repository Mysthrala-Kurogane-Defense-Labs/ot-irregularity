import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self,n_features,latent_dim=16):
        super().__init__();h1=max(8,min(128,n_features*4));h2=max(4,min(64,n_features*2));latent=max(2,min(latent_dim,h2))
        self.net=nn.Sequential(nn.Linear(n_features,h1),nn.ReLU(),nn.Dropout(.05),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,latent),nn.ReLU(),nn.Linear(latent,h2),nn.ReLU(),nn.Linear(h2,h1),nn.ReLU(),nn.Linear(h1,n_features))
    def forward(self,x):return self.net(x)

def train_ae(x,validation_x,cfg,seed,path):
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True,warn_only=True);model=Autoencoder(x.shape[1],cfg.get("latent_dim",16));opt=torch.optim.Adam(model.parameters(),lr=cfg.get("learning_rate",.001),weight_decay=1e-5);tensor=torch.tensor(x,dtype=torch.float32);validation=torch.tensor(validation_x,dtype=torch.float32);best=float("inf");patience=int(cfg.get("patience",10));stale=0;path=__import__("pathlib").Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for _ in range(int(cfg.get("epochs",100))):
        model.train();order=torch.randperm(len(tensor))
        for ids in order.split(int(cfg.get("batch_size",128))):
            batch=tensor[ids];opt.zero_grad();loss=((model(batch)-batch)**2).mean();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():score=((model(validation)-validation)**2).mean().item()
        if score<best:best=score;stale=0;torch.save({"state_dict":model.state_dict(),"n_features":x.shape[1],"latent_dim":cfg.get("latent_dim",16),"best_validation_loss":best},path)
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["state_dict"]);model.eval();return model

def load_ae(path):
    c=torch.load(path,map_location="cpu",weights_only=True);m=Autoencoder(c["n_features"],c["latent_dim"]);m.load_state_dict(c["state_dict"]);m.eval();return m

def ae_errors(model,x):
    with torch.no_grad():return (model(torch.tensor(x,dtype=torch.float32)).numpy()-x)**2
