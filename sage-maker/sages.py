import pytorch_lightning as zeus
import os
from PIL import Image
from alive_progress import alive_bar
from pathlib import Path
import multiprocessing
import urllib.request
import io
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader, random_split
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict


class Kaokore(Dataset):
    def __init__(
        self,
        data_dir,
        train=True,
        split="train",
        category="status",
        transform=None,
        metadata_dir="metadata",
        image_dir="sages",
    ):
        self.train = train  # use this to split up data
        self.data_dir = Path(data_dir)
        self.split = split
        self.category = category
        self.images = os.listdir(self.data_dir)
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_filename = self.images[index]

        with open(self.image_dir / image_filename, "rb") as f:
            image = Image.open(f).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class SagesDataModule(zeus.LightningDataModule):
    def __init__(
        self,
        image_dir="sages",
        metadata_dir="metadata",
        workers=16,
        download_threads=16,
        transform=None,
        batch_size=32,
    ):
        super().__init__()
        self.cwd = Path(os.getcwd())
        self.metadata_dir = self.cwd / metadata_dir
        self.images_dir = self.cwd / image_dir
        self.workers = workers
        self.download_threads = download_threads
        self.transform = transform
        self.batch_size = batch_size

    def download_image(self, url):
        index, url = url
        image_file = self.images_dir / f"{index}.jpg"

        image_data = urllib.request.urlopen(url).read()

        try:
            # convert grayscale to RGB
            img = Image.open(io.BytesIO(image_data))
            if img.mode != "RGB":
                pass
                # arr = np.asarray(img)
                # assert len(arr.shape) == 2
                # arr = np.stack([arr] * 3, axis=-1)
                # rgb_img = Image.fromarray(arr, mode="RGB")
                # rgb_img.save(image_file)
            else:
                open(image_file, "wb").write(image_data)
            img.close()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting early!")
            sys.exit(130)
        except Exception as e:
            print(f"Download failed with {e} for {index}-th image from {url}")

    def prepare_data(self):
        os.makedirs(self.images_dir, exist_ok=True)
        if len(os.listdir(self.images_dir)) > 8800:
            print("already downloaded data")
            return

        with open(self.metadata_dir / "urls.txt", "r") as f:
            urls = [(index, url) for index, url in enumerate(f.read().splitlines())]

        pool = multiprocessing.Pool(self.download_threads)

        with alive_bar(len(urls)) as bar:
            for i, _ in enumerate(pool.imap_unordered(self.download_image, urls)):
                bar()

    def setup(self, stage: Optional[str] = None):  # what does stage do?
        n_images = len(
            os.listdir(self.images_dir)
        )  # how does this work across devices?
        n_train = int(n_images * 0.8)
        n_val = n_images - n_train
        if stage == "fit" or stage is None:
            data_full = Kaokore(self.images_dir, train=True, transform=self.transform)
            self.sages_train, self.sages_val = random_split(
                data_full,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.sages_test = Kaokore(
                self.images_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.sages_train, batch_size=self.batch_size, num_workers=self.workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.sages_val, batch_size=self.batch_size, num_workers=self.workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.sages_test, batch_size=self.batch_size, num_workers=self.workers
        )


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(zeus.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim, img_shape=data_shape
        )
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


if __name__ == "__main__":
    tf = transforms.Compose([transforms.ToTensor()])
    data = SagesDataModule(download_threads=128, transform=tf)
    model = GAN(8, 3, 5)
    trainer = zeus.Trainer(max_epochs=5)
    trainer.fit(model, data)