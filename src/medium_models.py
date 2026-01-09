import torch
import torch.nn as nn

class HybridConvVAE(nn.Module):
    def __init__(self, audio_dim, lyric_dim, latent_dim=32):
        super(HybridConvVAE, self).__init__()
        
        
        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
    
        self.combined_dim = 16 * (audio_dim // 2) + lyric_dim
        
        self.fc_mu = nn.Linear(self.combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.combined_dim, latent_dim)
        
      
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, audio_dim + lyric_dim) 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, audio, lyrics):
       
        a_feat = self.audio_conv(audio.unsqueeze(1))
        
      
        combined = torch.cat((a_feat, lyrics), dim=1)
        
        mu, logvar = self.fc_mu(combined), self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)
        
        return self.decoder(z), mu, logvar