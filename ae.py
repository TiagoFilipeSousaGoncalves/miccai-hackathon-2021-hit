import torch
class AE(torch.nn.Module):
    def init(self):
        super().init()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),

        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 2, 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 2, 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, 2, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_embeds(self, x):
        global_avg_pool = torch.nn.AvgPool2d(16)
        enc = self.encoder(x)
        enc = global_avg_pool(enc)
        enc = torch.reshape(enc, (-1, 64))
        return enc