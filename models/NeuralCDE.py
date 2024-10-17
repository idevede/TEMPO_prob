import torch
import torch.nn as nn
import torchcde
import torch.nn.functional as F

# # Define the CDE function
# class CDEFunc(nn.Module):
#     def __init__(self, hidden_dim):
#         super(CDEFunc, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(hidden_dim, 50),
#             nn.Tanh(),
#             nn.Linear(50, hidden_dim)
#         )

#     def forward(self, t, z):
#         # Ensure z has a batch dimension
#         # import pdb; pdb.set_trace()
#         result = self.net(z)
#         print(result.shape)
#         return result
class CDEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(CDEFunc, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, t, z):
        return self.relu(self.linear(z)).unsqueeze(2)

# Define the main forecasting model
class NeuralCDETimeSeries(nn.Module):
    def __init__(self, configs, device):
        super(NeuralCDETimeSeries, self).__init__()
        self.input_dim = configs.enc_in
        self.hidden_dim = configs.d_model
        self.seq_len = configs.seq_len
        self.output_dim = configs.enc_in
        self.pred_len = configs.pred_len
                
        
        self.initial_linear = nn.Linear(self.input_dim , self.hidden_dim)
        self.cdefunc = CDEFunc(self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.prediction_length = self.pred_len

        # Output layers for Student's t-distribution parameters
        self.mu = nn.Linear(configs.pred_len, configs.pred_len)  # Mean
        self.sigma = nn.Linear(configs.pred_len, configs.pred_len)  # Scale (standard deviation)
        self.nu = nn.Linear(configs.pred_len, configs.pred_len)  # Degrees of freedom
    

    def forward(self, x, ii):
        # x shape is [B, L, C] where C=1
        B, L, C = x.size()
        
        # Encode the initial state
        x0 = self.initial_linear(x[:, 0, :self.input_dim])  # Shape: [B, hidden_dim]
        
        # Create the path
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)
        
        # Integrate CDE
        t_span = torch.linspace(0, 1, self.prediction_length)
        # import pdb; pdb.set_trace()
        # print(x0.shape)
        z = torchcde.cdeint(X=X, func=self.cdefunc, z0=x0, t=t_span)
        
        # z shape: [L_p, B, hidden_dim]
        # import pdb; pdb.set_trace()
        # z = z.permute(1, 0, 2)  # Shape: [B, L_p, hidden_dim]
        
        # Decode the output
        outputs = self.output_linear(z)
        
        
        x = outputs.permute(0, 2, 1) # [B, L, D] -> [B, D, L]
        # import pdb; pdb.set_trace()
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x)) + 1e-6  # Ensure scale is positive
        nu = F.softplus(self.nu(x)) + 2   # Ensure degrees of freedom > 2

        
        # Output shape: [B, L_p, C]
        return (mu, sigma, nu)