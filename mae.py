import numpy as np
import torch
import torch.nn as nn
from experimental.deepanshi.mae_reproduction.vit import Block
from loguru import logger
    

class MAE(nn.Module):
    def __init__(self,
        encoder_num_blocks,
        decoder_num_blocks,
        encoder_num_heads,
        decoder_num_heads,
        num_emb_encoder,
        num_emb_decoder,
        head_size,
        patch_size, # p
        input_size, # c, h, w
        device,
        ):
        super().__init__()
        c, h, w = input_size
        self.patch_size = patch_size
        self.num_patches = h * w // (patch_size ** 2)

        self.encoder = nn.Sequential(*[
            Block(encoder_num_heads,
            head_size,
            num_emb_encoder,
            ) for _ in range(encoder_num_blocks)
        ])
        self.encoder_norm = nn.LayerNorm(num_emb_encoder)

        self.decoder_embed = nn.Linear(num_emb_encoder, num_emb_decoder)

        self.decoder = nn.Sequential(*[
            Block(decoder_num_heads,
            head_size,
            num_emb_decoder,
            ) for _ in range(decoder_num_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(num_emb_decoder)
        self.decoder_pred = nn.Linear(num_emb_decoder, self.patch_size**2 * 3, bias=True) # decoder to patch

        input_dim = self.patch_size**2 * c
        self.patch_emb_layer = nn.Linear(input_dim, num_emb_encoder)
        self.encoder_positional_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_emb_encoder))
        self.decoder_positional_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_emb_decoder))
        

        self.mask_token = nn.Parameter(torch.zeros(1, 1, num_emb_decoder))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_emb_encoder))


        ### loss fn
        self.loss = nn.MSELoss()

        self.device = device

    def patchify(self, x):
        temp = int(np.sqrt(self.num_patches))
        patch_list = []

        for i in range(temp):
            for j in range(temp):
                curr = x[:, :, i*self.patch_size:(i+1)*self.patch_size,  j*self.patch_size:(j+1)*self.patch_size]
                patch_list.append(curr.flatten(1).unsqueeze(-1))

        patches = torch.cat(patch_list, -1).transpose(1, 2)
        return patches

    def masking_algo(self, input_embeddings):
        #input_embeddings [N, num_patches + 1, input_dim]
        batch_size, a, input_dim = input_embeddings.shape
        num_patches = a - 1
        cls_token, input_emb = input_embeddings[:, 0, :], input_embeddings[:, 1:, :]
        unshuffled_ind = np.arange(num_patches)

        np.random.shuffle(unshuffled_ind)
        unmasked_num_patches_len = int(0.25 * num_patches)
        unmasked_idx = unshuffled_ind[:unmasked_num_patches_len]
        masked_idx = unshuffled_ind[unmasked_num_patches_len:]

        unmasked_embeddings = input_emb[:, unmasked_idx, :]
        unmasked_embeddings = torch.cat([cls_token.unsqueeze(1), unmasked_embeddings], axis=1)
        return unmasked_embeddings, unshuffled_ind, masked_idx


    def unshuffle_embeddings(self, masked_embeddings, shuffled_ind):
        cls_token, masked_emb = masked_embeddings[:, 0, :], masked_embeddings[:, 1:, :]
        n, a, input_dim = masked_emb.shape

        shared_masked_emb = self.mask_token.expand(n, self.num_patches - a, input_dim)
        
        all_embeddings = torch.cat([masked_emb, shared_masked_emb], axis=1)
        
        shuffled_ind = np.argsort(shuffled_ind)
        unshuffled_emb = torch.gather(all_embeddings, 1, torch.tensor(shuffled_ind).unsqueeze(0).unsqueeze(-1).repeat(n, 1, input_dim).to(self.device))
        
        unshuffle_emb = torch.cat([cls_token.unsqueeze(1), unshuffled_emb], axis=1)
        return unshuffle_emb

    def loss_compute(self, inp_patches, op_patches, masked_idx):
        inp_patches_mean = torch.mean(inp_patches, -1, keepdim=True)
        inp_patches_var = torch.var(inp_patches, -1, keepdim=True)

        inp_patches = (inp_patches - inp_patches_mean)/(inp_patches_var + 1e-6)**0.5

        masked_inp_patches = inp_patches[:, masked_idx, :]
        masked_op_patches = op_patches[:, masked_idx, :]

        computed_loss = self.loss(masked_inp_patches, masked_op_patches)
        return computed_loss

    def forward(self, x:torch.tensor):
        # x : (N, C, H, W)
        #generate image patches
        image_patches = self.patchify(x)

        input_embedding = self.patch_emb_layer(image_patches)

        batch_size = input_embedding.shape[0]

        #add cls_token and positional embeddings
        cls_token = self.cls_token.expand([batch_size, -1, -1])
        input_embedding = torch.cat([cls_token, input_embedding], dim=1)

        #add positional embeddings
        input_embedding += self.encoder_positional_emb

        #mask embedidngs
        unmasked_embedding, shuffled_idx, masked_idx = self.masking_algo(input_embedding)

        #call encoder
        encoded_embeddings = self.encoder(unmasked_embedding)
        encoded_embeddings = self.encoder_norm(encoded_embeddings)
        encoded_embeddings = self.decoder_embed(encoded_embeddings)

        #unmask
        unshuffled_emb = self.unshuffle_embeddings(encoded_embeddings, shuffled_idx)

        unshuffled_emb += self.decoder_positional_emb
        #call decoder
        decoder_embeddings = self.decoder(unshuffled_emb)
        decoder_embeddings = self.decoder_norm(decoder_embeddings)
        output_ = self.decoder_pred(decoder_embeddings)
        output_ = output_[:, 1:, :]

        #loss compute
        batch_loss = self.loss_compute(image_patches, output_, masked_idx)

        return batch_loss, output_
    
