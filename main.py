# # Minimal FID computation
from flax.training.common_utils import shard
from flax.jax_utils import replicate
import jax
import jax.numpy as jnp
import numpy as np
import functools
import torch
from inception import fid_score, InceptionV3
from tqdm.auto import tqdm
import datasets
from torchvision import transforms
import os

# Model
model = InceptionV3(pretrained=True)
H, C = 512, 3 # Training resolution
fid_fn = functools.partial(model.apply, train=False)
fid_fn_p = jax.pmap(fid_fn)
key = jax.random.PRNGKey(104)
init_params = model.init(key, jnp.ones((1, H, H, C)))
init_params_p = replicate(init_params)

# Data
dataset = datasets.load_dataset("cifar10")
train_transforms = transforms.Compose(
    [
        transforms.Resize(H, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
def preprocess_fid(examples):
    images = examples['img']
    examples["pixel_values"] = [train_transforms(image) for image in images]
    return examples

def fid_collate_fn(examples):
    return torch.stack([example["pixel_values"] for example in examples]).numpy()

fid_dataset = dataset["train"].with_transform(preprocess_fid)
sampler = torch.utils.data.RandomSampler(fid_dataset, replacement=True, num_samples=int(1e10))
num_samples = jax.device_count()
fid_loader = torch.utils.data.DataLoader(fid_dataset, batch_size=num_samples, collate_fn=fid_collate_fn, sampler=sampler)

# FID stats
procs = []
fid_steps = 5000
for i, x in tqdm(enumerate(fid_loader),desc="Calculating FID stats...",total=fid_steps):
    if i >= fid_steps:
        break
    x = np.moveaxis(x,-3,-1) # (8,H,H,3) -1<=x<=1
    x = jax.image.resize(x,shape=(num_samples, 256, 256, C),method="bicubic")  # 256x256 as is in most FID implementations
    x = shard(x)
    proc = fid_fn_p(init_params_p, jax.lax.stop_gradient(x))
    procs.append(proc.squeeze(axis=1).squeeze(axis=1).squeeze(axis=1))

# Checkpoint
precomputed_fid_stats = False
stats_path = os.path.join('.', 'fid_stats.npz')
if os.path.isfile(stats_path) and precomputed_fid_stats:
    stats = np.load(stats_path)
    mu0, sigma0 = stats["mu"], stats["sigma"]
    print('Loaded pre-computed statistics at:', stats_path)
np.savez(stats_path, mu=mu0, sigma=sigma0)
print('Saved pre-computed statistics at:', stats_path, '. Set --precomputed_fid_stats=True to skip it next time!')

procs = jnp.concatenate(procs, axis=0)
mu0 = np.mean(procs, axis=0)
sigma0 = np.cov(procs, rowvar=False)

# During Training with diffusers:
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
# rng = jax.random.PRNGKey(123)
# fid_bar = tqdm(desc="Computing FID stats...", total=fid_steps)
# procs = []
# for i in range(fid_steps): # fid_steps * 8 (num_devices) samples
#     rng, key = jax.random.split(rng)
#     keys = jax.random.split(key, num_samples)
#     images = pipeline(prompt_ids, params, keys, num_inference_steps,height=H,width=H,jit=True).images # on-device (8,1,H,H,3) 
#     proc = fid_fn_p(init_params_p, jax.lax.stop_gradient(2 * images - 1)) # Inception-Net States
#     procs.append(proc.squeeze(axis=1).squeeze(axis=1).squeeze(axis=1))
#     fid_bar.update(1)

# procs = jnp.concatenate(procs, axis=0)
# mu = np.mean(procs, axis=0)
# sigma = np.cov(procs, rowvar=False)

# fid_score = inception.fid_score(mu0,mu,sigma0,sigma)

# writer.add_scalar("FID/train", fid_score, global_step)
# del procs, images
# fid_bar.close()

# Example FID Score
mu = mu0 + .1 # try: + np.random.random(mu0.shape) *.01
sigma = sigma0 * 1.1 # try: + np.random.random(sigma0.shape) *.01

fid_score(mu0, mu, sigma0, sigma)