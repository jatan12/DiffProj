import torch 
import torch.nn as nn 

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# Prevents NaN by torch.log(0)
def torch_log(x):
    return torch.log(torch.clamp(x, min = 1e-10))

# Encoder
class Encoder(nn.Module):
    def __init__(self, inp_dim, c_bar_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
                
        # Encoder Architecture
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim + c_bar_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Mean and Variance
        self.mu = nn.Linear(256, z_dim)
        self.var = nn.Linear(256, z_dim)
        
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        out = self.encoder(x)
        mu = self.mu(out)
        var = self.var(out)
        return mu, self.softplus(var)
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, inp_dim, c_bar_dim, hidden_dim, z_dim):
        super(Decoder, self).__init__()
        
        # Decoder Architecture
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + inp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, c_bar_dim)
        )
    
    def forward(self, x):
        out = self.decoder(x)
        return out

class Beta_cVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(Beta_cVAE, self).__init__()
        
        # Encoder & Decoder
        self.encoder = encoder
        self.decoder = decoder
        
        # RCL Loss
        self.rcl_loss = nn.MSELoss()
            
    # Encoder
    def _encoder(self, x, y):
        inputs = torch.cat([x, y], dim = 1)
        mean, std = self.encoder(inputs)        
        return mean, std

    # Decoder without DDN / Optimization Layer
    def _decoder(self, z, x):
        inputs = torch.cat([z, x], dim = 1)
        y = self.decoder(inputs) # Behavioral params
        return y
            
# Get the Weights
def get_params(four_lane=True, dir=""):
    
    # Beta-cVAE Inputs
    enc_inp_dim = 55
    enc_out_dim = 200 # 200 / 2
    dec_inp_dim = enc_inp_dim
    dec_out_dim = 30
    hidden_dim = 1024 * 2
    z_dim = 2

    # Load the Trained Model
    encoder = Encoder(enc_inp_dim, enc_out_dim, hidden_dim, z_dim)
    decoder = Decoder(dec_inp_dim, dec_out_dim, hidden_dim, z_dim)
    model = Beta_cVAE(encoder, decoder)
    if four_lane:
        model.load_state_dict(torch.load(dir)) 
    else:
        model.load_state_dict(torch.load(dir))
    model.eval()
    
    # Extracting the weights & biases
    W0 = model.state_dict()['decoder.decoder.0.weight'].detach().numpy()
    b0 = model.state_dict()['decoder.decoder.0.bias'].detach().numpy()

    W1 = model.state_dict()['decoder.decoder.3.weight'].detach().numpy()
    b1 = model.state_dict()['decoder.decoder.3.bias'].detach().numpy()

    W2 = model.state_dict()['decoder.decoder.6.weight'].detach().numpy()
    b2 = model.state_dict()['decoder.decoder.6.bias'].detach().numpy()

    W3 = model.state_dict()['decoder.decoder.9.weight'].detach().numpy()
    b3 = model.state_dict()['decoder.decoder.9.bias'].detach().numpy()

    W4 = model.state_dict()['decoder.decoder.12.weight'].detach().numpy()
    b4 = model.state_dict()['decoder.decoder.12.bias'].detach().numpy()

    W5 = model.state_dict()['decoder.decoder.15.weight'].detach().numpy()
    b5 = model.state_dict()['decoder.decoder.15.bias'].detach().numpy()

    Wandb = [W0, b0, 
            W1, b1, 
            W2, b2, 
            W3, b3, 
            W4, b4, 
            W5, b5]
    
    # Batch Norm Parameters
    scale_0 = model.state_dict()['decoder.decoder.1.weight'].detach().numpy()
    bias_0 = model.state_dict()['decoder.decoder.1.bias'].detach().numpy()
    mean_0 = model.state_dict()['decoder.decoder.1.running_mean'].detach().numpy()
    var_0 = model.state_dict()['decoder.decoder.1.running_var'].detach().numpy()

    scale_1 = model.state_dict()['decoder.decoder.4.weight'].detach().numpy()
    bias_1 = model.state_dict()['decoder.decoder.4.bias'].detach().numpy()
    mean_1 = model.state_dict()['decoder.decoder.4.running_mean'].detach().numpy()
    var_1 = model.state_dict()['decoder.decoder.4.running_var'].detach().numpy()

    scale_2 = model.state_dict()['decoder.decoder.7.weight'].detach().numpy()
    bias_2 = model.state_dict()['decoder.decoder.7.bias'].detach().numpy()
    mean_2 = model.state_dict()['decoder.decoder.7.running_mean'].detach().numpy()
    var_2 = model.state_dict()['decoder.decoder.7.running_var'].detach().numpy()

    scale_3 = model.state_dict()['decoder.decoder.10.weight'].detach().numpy()
    bias_3 = model.state_dict()['decoder.decoder.10.bias'].detach().numpy()
    mean_3 = model.state_dict()['decoder.decoder.10.running_mean'].detach().numpy()
    var_3 = model.state_dict()['decoder.decoder.10.running_var'].detach().numpy()

    scale_4 = model.state_dict()['decoder.decoder.13.weight'].detach().numpy()
    bias_4 = model.state_dict()['decoder.decoder.13.bias'].detach().numpy()
    mean_4 = model.state_dict()['decoder.decoder.13.running_mean'].detach().numpy()
    var_4 = model.state_dict()['decoder.decoder.13.running_var'].detach().numpy()

    BN = [scale_0, bias_0, mean_0, var_0, 
          scale_1, bias_1, mean_1, var_1, 
          scale_2, bias_2, mean_2, var_2, 
          scale_3, bias_3, mean_3, var_3, 
          scale_4, bias_4, mean_4, var_4]
    
    return Wandb, BN