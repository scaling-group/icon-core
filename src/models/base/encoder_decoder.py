#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder_in_proj: nn.Module,
        decoder_in_proj: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        out_proj: nn.Module,
    ):
        super().__init__()
        self.encoder_in_proj = encoder_in_proj
        self.decoder_in_proj = decoder_in_proj
        self.encoder = encoder
        self.decoder = decoder
        self.out_proj = out_proj

    def forward(self, memory, query):
        memory = self.encoder_in_proj(memory)
        query = self.decoder_in_proj(query)
        x = self.encoder(memory)
        x = self.decoder(query, x)
        x = self.out_proj(x)
        return x
