import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64, dropout=0.1):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2, dropout)
        self.enc3 = DoubleConv(b * 2, b * 4, dropout)
        self.enc4 = DoubleConv(b * 4, b * 8, dropout)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(b * 8, b * 16, dropout)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8, dropout)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4, dropout)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2, dropout)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.out_conv = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class DrivableSegmentor:
    """Wrapper for U-Net inference — returns binary drivable mask."""

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 img_w: int = 512, img_h: int = 256, base: int = 64):
        self.device = device
        self.img_w = img_w
        self.img_h = img_h

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model_type = config.get("model_type", "unet_scratch")

        if model_type == "smp_resnet18":
            import segmentation_models_pytorch as smp
            self.model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=3,
                classes=2,
                activation=None,
            )
        else:
            self.model = UNet(in_channels=3, num_classes=2, base=base, dropout=0.1)

        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_bgr: OpenCV BGR image (H, W, 3)
        Returns:
            Binary mask (H, W) uint8 with 255=drivable, 0=non-drivable
            at the model's resolution (img_w x img_h).
        """
        rgb = Image.fromarray(frame_bgr[:, :, ::-1])
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type="cuda", enabled=self.device == "cuda"):
            logits = self.model(tensor)

        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8) * 255
        return mask
