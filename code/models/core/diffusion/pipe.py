from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps

class Pipe:
    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        self.scheduler = DDPMScheduler() if scheduler is None else scheduler
        self.device = device
        
    def train(self, dataloader, num_epochs=10, learning_rate=1e-4):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=500, num_training_steps=(len(dataloader) * num_epochs))
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch.keys() else None
                h_embeds = batch['h_embedding'].to(device)
                N = h_embeds.shape[0]

                if torch.rand(1) < 0.1:
                    c_embeds = None

                noise = torch.randn_like(h_embeds)
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)
                perturbed_h_embeds = self.scheduler.add_noise( h_embeds, noise, timesteps)
                noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
                loss = criterion(noise_pre, noise)
                loss = (loss).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                lr_scheduler.step()
                optimizer.step()
                loss_sum += loss.item()

            loss_epoch = loss_sum / len(dataloader)
            print(f'epoch: {epoch}, loss: {loss_epoch}')

    def generate(self, c_embeds=None, num_inference_steps=50, timesteps=None, guidance_scale=5.0, generator=None):
        self.diffusion_prior.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)
        h_t = torch.randn(N, self.diffusion_prior.embed_dim, generator=generator, device=self.device)
        
        for _, t in tqdm(enumerate(timesteps)):
            # Get the scalar timestep value (same for all samples in batch)
            t_scalar = t.item() if torch.is_tensor(t) else t
            
            # Create tensor of timesteps for the model (one per batch element)
            t_tensor = torch.ones(h_t.shape[0], dtype=torch.float, device=self.device) * t_scalar
            
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.diffusion_prior(h_t, t_tensor)
            else:
                noise_pred_cond = self.diffusion_prior(h_t, t_tensor, c_embeds)
                noise_pred_uncond = self.diffusion_prior(h_t, t_tensor)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Use scalar timestep for scheduler.step()
            h_t = self.scheduler.step(noise_pred, t_scalar, h_t, generator=generator).prev_sample
        
        return h_t