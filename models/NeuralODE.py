import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F

# Define the ODE function
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim)
        )

    def forward(self, t, x):
        return self.net(x)

# Define the ODE Block
class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, x, t_span):
        # Integrate ODE
        out = odeint(self.odefunc, x, t_span)
        return out

# Define the main forecasting model
class NeuralODETimeSeries(nn.Module):
    def __init__(self, configs, device):
        self.input_dim = configs.enc_in
        self.hidden_dim = configs.d_model
        self.seq_len = configs.seq_len
        self.output_dim = configs.enc_in
        self.pred_len = configs.pred_len
       
        super(NeuralODETimeSeries, self).__init__()
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.odefunc = ODEFunc(self.hidden_dim)
        self.odeblock = ODEBlock(self.odefunc)
        self.decoder = nn.Linear(self.hidden_dim, configs.enc_in)
        self.prediction_length = self.pred_len

        # Output layers for Student's t-distribution parameters
        self.mu = nn.Linear(configs.pred_len, configs.pred_len)  # Mean
        self.sigma = nn.Linear(configs.pred_len, configs.pred_len)  # Scale (standard deviation)
        self.nu = nn.Linear(configs.pred_len, configs.pred_len)  # Degrees of freedom
    

    def forward(self, x, ii):
        # Assume x shape is [B, L, C] where C=1
        B, L, C = x.size() 
        
        # Encode input
        x = x.view(B * L, C)
        h = self.encoder(x)
        
        # Reshape for ODE integration
        h = h.view(B, L, -1)
        # import pdb; pdb.set_trace()
        t_span = torch.linspace(0, 1, self.prediction_length)
        
        # Integrate ODE
        h_out = self.odeblock(h[:, -1, :], t_span)
        # import pdb; pdb.set_trace()
        # Decode output
        # h_out = h_out[-1]  # Take the last time step
        # h_out = h_out.view(B * self.prediction_length, -1)
        # output = self.decoder(h_out)

        # h_out shape: [L_p, B, hidden_dim]
        h_out = h_out.permute(1, 0, 2)  # Shape: [B, L_p, hidden_dim]
        
        # Decode output
        output = self.decoder(h_out)
        
        # Reshape output
        outputs = output.view(B, self.prediction_length, C)
        # import pdb; pdb.set_trace()
        x = outputs.permute(0, 2, 1) # [B, L, D] -> [B, D, L]
        # import pdb; pdb.set_trace()
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x)) + 1e-6  # Ensure scale is positive
        nu = F.softplus(self.nu(x)) + 2   # Ensure degrees of freedom > 2


        # if self.pool:
        #     return outputs, loss_local #loss_local - reduce_sim_trend - reduce_sim_season - reduce_sim_noise
        return (mu, sigma, nu)


# # Example usage
# input_dim = 1  # Each time step has a single feature
# hidden_dim = 32
# output_dim = 1
# prediction_length = 10

# model = NeuralODETimeSeries(input_dim, hidden_dim, output_dim, prediction_length)
# x = torch.randn(8, 20, 1)  # Example input [B=8, L=20, C=1]
# output = model(x)
# print(output.shape)  # Expected output shape: [8, 10, 1]