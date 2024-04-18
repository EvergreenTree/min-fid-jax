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

# Model
model = inception.InceptionV3(pretrained=True)
H, C = 256, 3
fid_fn = functools.partial(model.apply, train=False)
fid_fn_p = jax.pmap(fid_fn)
key = jax.random.PRNGKey(104)
init_params = model.init(key, jnp.ones((1, H, H, C)))
init_params_p = replicate(init_params)

# Data
dataset = datasets.load_dataset("cifar10")
train_transforms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
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
    x = np.moveaxis(x,-3,-1) # (8,512,512,3) -1<=x<=1
    x = shard(x)
    proc = fid_fn_p(init_params_p, jax.lax.stop_gradient(x))
    procs.append(proc.squeeze(axis=1).squeeze(axis=1).squeeze(axis=1))
procs = jnp.concatenate(procs, axis=0)
mu0 = np.mean(procs, axis=0)
sigma0 = np.cov(procs, rowvar=False)

# FID Score
fid_score(mu0, mu0, sigma0, sigma0)
