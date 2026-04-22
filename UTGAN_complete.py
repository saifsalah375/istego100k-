"""
UT-GAN Complete Implementation
================================
Paper: "An Embedding Cost Learning Framework Using GAN"
J. Yang, D. Ruan, J. Huang, X. Kang and Y. Shi
IEEE Transactions on Information Forensics and Security, vol. 15, pp. 839-851, 2020.

Unofficial PyTorch Implementation — All files merged into one.
"""

# ─────────────────────────────────────────────
#  SECTION 1: IMPORTS
# ─────────────────────────────────────────────
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.autograd import Variable
from torchvision.utils import make_grid
from glob import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import math
import random
import csv
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # غيّر هذا حسب GPU المتاحة


# ─────────────────────────────────────────────
#  SECTION 2: MODULE.PY — Building Blocks
# ─────────────────────────────────────────────

class ABS(nn.Module):
    """دالة القيمة المطلقة كطبقة"""
    def forward(self, x):
        return torch.abs(x)


class LRelu(nn.Module):
    """Leaky ReLU مخصص"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.relu_(x) - self.alpha * F.relu_(-x)


class DoubleConv(nn.Module):
    """
    طبقتا Convolution متتاليتان مع BatchNorm.
    خيارات التفعيل: ReLU أو LReLU أو بدون.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, relu=False, lrelu=False):
        super().__init__()
        if relu:
            act = nn.ReLU(inplace=True)
        elif lrelu:
            act = LRelu()
        else:
            act = nn.Identity()

        self.double_conv = nn.Sequential(
            act if (relu or lrelu) else nn.Identity(),
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if relu else (LRelu() if lrelu else nn.Identity()),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """Conv بـ stride=2 للتصغير"""
    def __init__(self, in_ch, out_ch, kernel_size=3, relu=False, lrelu=False):
        super().__init__()
        act_pre = nn.ReLU(inplace=True) if relu else (LRelu() if lrelu else nn.Identity())
        self.down_conv = nn.Sequential(
            act_pre,
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.down_conv(x)


class Down(nn.Module):
    """
    طبقة التصغير (Encoder block).
    inc=True → بدون دالة تفعيل قبل Conv (أول طبقة في الشبكة).
    """
    def __init__(self, in_ch, out_ch, lrelu=True, inc=False):
        super().__init__()
        if inc:
            self.conv_block = DownConv(in_ch, out_ch, relu=False, lrelu=False)
        elif lrelu:
            self.conv_block = DownConv(in_ch, out_ch, lrelu=True)
        else:
            self.conv_block = DownConv(in_ch, out_ch, relu=True)

    def forward(self, x):
        return self.conv_block(x)


class Up(nn.Module):
    """
    طبقة التكبير (Decoder block) مع Skip Connection.
    bilinear=True  → Upsample + DoubleConv
    bilinear=False → ConvTranspose2d (الافتراضي في هذا النموذج)
    """
    def __init__(self, in_ch, out_ch, bilinear=True, dropout=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DoubleConv(in_ch, out_ch, relu=True),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.up = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch, out_ch // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch // 2),
            )
        self.dp_fg = dropout
        self.dp = nn.Dropout(p=0.5)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.dp_fg:
            x1 = self.dp(x1)
        # تعديل الأبعاد لضمان التطابق مع Skip Connection
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], dim=1)


class OutConv(nn.Module):
    """
    طبقة الإخراج النهائية للـ Generator.
    تُنتج خريطة الاحتمالية (Probability Map) في [0, 0.5].
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_ch, out_ch, relu=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.sigmoid(x) - 0.5
        x = self.relu(x)      # القيم في [0, 0.5]
        return x


# ─── High-Pass Filters (SRM Kernels) ───
HPF = np.zeros([6, 1, 5, 5], dtype=np.float32)
HPF[0,0] = [[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]]
HPF[1,0] = [[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0]]
HPF[2,0] = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
HPF[3,0] = [[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]]
HPF[4,0] = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
HPF[5,0] = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]


class HPFConv2d(nn.Module):
    """
    فلاتر SRM ثابتة (غير قابلة للتدريب).
    تستخرج الضوضاء عالية التردد من الصورة.
    """
    def __init__(self, in_channels=1, out_channels=6):
        super().__init__()
        self.hpf_conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=5, padding=2, bias=False)
        self.hpf_conv.weight = nn.Parameter(
            torch.tensor(HPF), requires_grad=False)

    def forward(self, x):
        return self.hpf_conv(x)


class ConvTanBlock(nn.Module):
    """
    Conv → [ABS] → BN → Tanh → AvgPool
    المستخدمة في الطبقات الأولى من الـ Discriminator.
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1,
                 pool_size=5, pool_stride=2, abs=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2)]
        if abs:
            layers.append(ABS())
        layers += [
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.AvgPool2d(pool_size, pool_stride),
        ]
        self.block = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)


class ConvReluBlock(nn.Module):
    """
    Conv → BN → ReLU → [AdaptiveAvgPool | AvgPool]
    المستخدمة في الطبقات العميقة من الـ Discriminator.
    """
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1,
                 pool_size=5, pool_stride=2, f_output=False):
        super().__init__()
        pool = nn.AdaptiveAvgPool2d((16, 16)) if f_output else \
               nn.AvgPool2d(pool_size, pool_stride)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            pool,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
#  SECTION 3: GEN_MODELS.PY — Generator (U-Net)
# ─────────────────────────────────────────────

class Generator_Unet(nn.Module):
    """
    مولّد الـ UT-GAN بنيته معتمدة على U-Net.

    المدخل  : صورة cover بقناة واحدة (grayscale) بحجم 256x256.
    المخرج  : خريطة احتمالية (prob map) بنفس الحجم، قيمها في [0, 0.5].
              كل pixel يمثل احتمال تعديله أثناء عملية الإخفاء.

    الهندسة:
      Encoder: 8 طبقات تصغير (inc + down1..down7)
      Decoder: 7 طبقات تكبير مع Skip Connections (up1..up7)
      OutConv: تكبير أخير + Sigmoid - 0.5 + ReLU
    """
    def __init__(self, input_nc=1, out_nc=1, bilinear=False):
        super().__init__()
        self.bil = bilinear

        # ── Encoder ──
        self.inc   = Down(input_nc, 16,  inc=True)   # (B,16,128,128)
        self.down1 = Down(16,  32)                    # (B,32,64,64)
        self.down2 = Down(32,  64)                    # (B,64,32,32)
        self.down3 = Down(64,  128)                   # (B,128,16,16)
        self.down4 = Down(128, 128)                   # (B,128,8,8)
        self.down5 = Down(128, 128)                   # (B,128,4,4)
        self.down6 = Down(128, 128)                   # (B,128,2,2)
        self.down7 = Down(128, 128)                   # (B,128,1,1)

        factor = 2 if bilinear else 1

        # ── Decoder ──
        self.up1 = Up(128, 256 // factor, bilinear, dropout=True)   # →(B,128,2,2)
        self.up2 = Up(256, 256 // factor, bilinear, dropout=True)   # →(B,128,4,4)
        self.up3 = Up(256, 256 // factor, bilinear, dropout=True)   # →(B,128,8,8)
        self.up4 = Up(256, 256 // factor, bilinear)                  # →(B,128,16,16)
        self.up5 = Up(256, 128 // factor, bilinear)                  # →(B,64,32,32)
        self.up6 = Up(128, 64  // factor, bilinear)                  # →(B,32,64,64)
        self.up7 = Up(64,  32  // factor, bilinear)                  # →(B,16,128,128)

        self.outc = OutConv(32, out_nc)                              # →(B,1,256,256)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        x = self.up1(x8, x7)
        x = self.up2(x,  x6)
        x = self.up3(x,  x5)
        x = self.up4(x,  x4)
        x = self.up5(x,  x3)
        x = self.up6(x,  x2)
        x = self.up7(x,  x1)
        return self.outc(x)


# ─────────────────────────────────────────────
#  SECTION 4: DIS_MODELS.PY — Discriminator
# ─────────────────────────────────────────────

class Discriminator_steg(nn.Module):
    """
    مُميِّز الـ UT-GAN مبني على شبكة SRM-CNN.

    المدخل  : صورة بقناة واحدة (cover أو stego).
    المخرج  : logits لتصنيفين → [0: cover, 1: stego].

    الهندسة:
      HPFConv2d  → 6  خرائط بمرشحات SRM ثابتة
      ConvTan×2  → استخراج الضوضاء
      ConvRelu×3 → تعميق الميزات
      FC         → 128×16×16 → 2
    """
    def __init__(self, img_nc=1):
        super().__init__()
        self.Dis = nn.Sequential(
            HPFConv2d(img_nc, 6),
            ConvTanBlock(6,   8,   abs=True),
            ConvTanBlock(8,   16),
            ConvReluBlock(16, 32),
            ConvReluBlock(32, 64),
            ConvReluBlock(64, 128, f_output=True),
        )
        self.FC = nn.Linear(128 * 16 * 16, 2)

    def forward(self, x):
        feat = self.Dis(x).view(x.size(0), -1)
        return self.FC(feat)


# ─────────────────────────────────────────────
#  SECTION 5: DATALOADER.PY
# ─────────────────────────────────────────────

class MyDataset(TorchDataset):
    """
    محمّل البيانات الأساسي.
    يقرأ صور الـ cover من مجلد واحد ويُحوّلها.
    """
    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.cover_dir = dataset_dir
        self.cover_list = [
            os.path.basename(x)
            for x in glob(os.path.join(dataset_dir, '*'))
        ]
        assert len(self.cover_list) > 0, f"المجلد فارغ: {dataset_dir}"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        path = os.path.join(self.cover_dir, self.cover_list[idx])
        img = Image.open(path).convert('L')           # grayscale
        if self.transform:
            img = self.transform(img)
        return {'img': img}


class StegoDataset(TorchDataset):
    """
    محمّل بيانات موسّع لتوليد الـ dataset مع حفظ أسماء الملفات.
    مستخدم في مرحلة التوليد (Inference).
    """
    def __init__(self, img_dir, img_names, transform=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'img': img, 'name': self.img_names[idx]}


# ─────────────────────────────────────────────
#  SECTION 6: UTILS.PY
# ─────────────────────────────────────────────

def show_result(epoch, images, save=False, path='result.png'):
    """حفظ شبكة من الصور أثناء التدريب."""
    n = int(np.sqrt(images.size(0)))
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    for i, j in itertools.product(range(n), range(n)):
        axes[i, j].axis('off')
    for k in range(n * n):
        i, j = k // n, k % n
        axes[i, j].cla()
        img_data = images[k].cpu().data.numpy().transpose(1, 2, 0)
        if images.size(1) == 1:
            axes[i, j].imshow((img_data.squeeze() + 1) / 2, cmap='gray')
        else:
            axes[i, j].imshow((img_data + 1) / 2)
    fig.text(0.5, 0.04, f'Epoch {epoch}', ha='center')
    if save:
        plt.savefig(path)
    plt.close()


def _save_image_tif(tensor, filename, nrow=8):
    """حفظ tensor كصورة TIF."""
    grid = make_grid(tensor, nrow=nrow, normalize=True)
    arr = grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).byte().numpy()
    Image.fromarray(arr).convert('L').save(filename)


def imsave_single(tensor, path):
    """حفظ صورة واحدة من مخرجات الـ Generator."""
    _save_image_tif(tensor, path, nrow=1)


def calculate_psnr(img1, img2):
    """حساب PSNR بين صورتين في نطاق [0,1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    """حساب SSIM بين صورتين (CPU tensors بأبعاد C,H,W)."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        a = img1.cpu().permute(1, 2, 0).numpy()
        b = img2.cpu().permute(1, 2, 0).numpy()
        return ssim_fn(a, b, data_range=1.0, channel_axis=2)
    except ImportError:
        return -1.0


# ─────────────────────────────────────────────
#  SECTION 7: UTGAN.PY — Training Framework
# ─────────────────────────────────────────────

MODELS_PATH      = './models/'
TRAIN_RESULT_PATH = './train_result/'
EVAL_RESULT_PATH  = './eval_result/'


def weights_init(net):
    """تهيئة أوزان الشبكة باستخدام Xavier للـ Conv وNormal للـ BatchNorm."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            m.bias.requires_grad = False


class UTGAN:
    """
    إطار التدريب الكامل لـ UT-GAN.

    المكونات:
      netG     : Generator (U-Net) — يُنتج خريطة احتمالية
      netDisc  : Discriminator (SRM-CNN) — يُميّز cover عن stego
      criterion: CrossEntropyLoss

    دوال الخسارة:
      Loss_D = CrossEntropy(D(cover||stego), labels)
      Loss_G = -D_LAMBDA * Loss_D  +  Payld_LAMBDA * ||H(prob) - cap||²
    """

    def __init__(self, device, img_nc=1, lr=1e-4, payld=0.4, bilinear=False):
        self.device  = device
        self.img_nc  = img_nc
        self.lr      = lr
        self.payld   = payld
        self.bilinear = bilinear

        self.netG    = Generator_Unet(img_nc, bilinear=bilinear).to(device)
        self.netDisc = Discriminator_steg(img_nc).to(device)

        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        self.criterion   = nn.CrossEntropyLoss().to(device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr)

        for p in [MODELS_PATH, TRAIN_RESULT_PATH, EVAL_RESULT_PATH]:
            os.makedirs(p, exist_ok=True)

    # ── دالة تدريب دفعة واحدة ──
    def train_batch(self, cover, TANH_LAMBDA=60, D_LAMBDA=1.0, Payld_LAMBDA=1e-7):
        img_size   = cover.shape[2]
        batch_size = cover.shape[0]

        # ─── Noise عشوائي ───
        noise = torch.from_numpy(
            np.random.rand(*cover.shape)
        ).float().to(self.device)

        # ─── Labels ───
        label_zeros = np.zeros(batch_size)
        label_ones  = np.ones(batch_size)
        label = torch.from_numpy(
            np.stack([label_zeros, label_ones])
        ).long().view(-1).to(self.device)

        # ══ تدريب الـ Discriminator ══
        self.optimizer_D.zero_grad()
        with torch.no_grad():
            prob_pred = self.netG(cover)
        modi  = 0.5 * (torch.tanh((prob_pred + 2 * noise - 2) * TANH_LAMBDA)
                      - torch.tanh((prob_pred - 2 * noise) * TANH_LAMBDA))
        stego = (cover * 255 + modi) / 255.0

        data  = torch.cat([cover, stego], dim=0)
        pred_D = self.netDisc(data.detach())
        loss_D = self.criterion(pred_D, label)
        loss_D.backward()
        self.optimizer_D.step()

        # ══ تدريب الـ Generator ══
        self.optimizer_G.zero_grad()
        prob_pred = self.netG(cover)
        modi  = 0.5 * (torch.tanh((prob_pred + 2 * noise - 2) * TANH_LAMBDA)
                      - torch.tanh((prob_pred - 2 * noise) * TANH_LAMBDA))
        stego = (cover * 255 + modi) / 255.0

        data   = torch.cat([cover, stego], dim=0)
        pred_D = self.netDisc(data)
        loss_D = self.criterion(pred_D, label)

        # ─── Payload Entropy Loss ───
        eps = 1e-5
        p_plus  = prob_pred / 2.0 + eps
        p_minus = prob_pred / 2.0 + eps
        p_unch  = 1.0 - prob_pred + eps

        cap_entropy = torch.sum(
            -p_plus  * torch.log2(p_plus)
            -p_minus * torch.log2(p_minus)
            -p_unch  * torch.log2(p_unch),
            dim=(1, 2, 3)
        )
        cap = img_size * img_size * self.payld
        loss_entropy = torch.mean(torch.pow(cap_entropy - cap, 2))

        loss_G = D_LAMBDA * (-loss_D) + Payld_LAMBDA * loss_entropy
        loss_G.backward()
        self.optimizer_G.step()

        return loss_D.item(), loss_G.item()

    # ── حلقة التدريب الكاملة ──
    def train(self, train_loader, epochs):
        sample = next(iter(train_loader))
        data_fixed  = Variable(sample['img'].to(self.device))
        noise_fixed = torch.from_numpy(
            np.random.rand(*data_fixed.shape)
        ).float().to(self.device)

        for epoch in range(1, epochs + 1):
            loss_D_sum = loss_G_sum = 0.0

            for batch in train_loader:
                imgs = batch['img'].to(self.device)
                d, g = self.train_batch(imgs)
                loss_D_sum += d
                loss_G_sum += g

            n = len(train_loader)
            print(f"Epoch {epoch:4d} | loss_D: {loss_D_sum/n:.6f} | "
                  f"loss_G: {loss_G_sum/n:.6f}")

            # حفظ صور التدريب كل epoch
            with torch.no_grad():
                prob_f = self.netG(data_fixed)
                modi_f = 0.5 * (
                    torch.tanh((prob_f + 2 * noise_fixed - 2) * 60)
                    - torch.tanh((prob_f - 2 * noise_fixed) * 60)
                )
                stego_f = (data_fixed * 255 + modi_f) / 255.0
            show_result(epoch, prob_f,       save=True, path=TRAIN_RESULT_PATH+f'{epoch}_prob.png')
            show_result(epoch, stego_f,      save=True, path=TRAIN_RESULT_PATH+f'{epoch}_stego.png')
            show_result(epoch, data_fixed,   save=True, path=TRAIN_RESULT_PATH+f'{epoch}_cover.png')

            # حفظ الموديل كل 100 epoch
            if epoch % 100 == 0:
                torch.save(self.netG.state_dict(),
                           MODELS_PATH + f'netG_epoch_{epoch}.pth')
                print(f"  ✔ تم حفظ الموديل في: {MODELS_PATH}netG_epoch_{epoch}.pth")


# ─────────────────────────────────────────────
#  SECTION 8: DATASET GENERATION (Inference)
#  توليد الـ IStego100K++ Dataset
# ─────────────────────────────────────────────

def generate_stego_dataset(
    model_path,
    data_path,
    output_folder,
    n_images=100,
    img_size=1024,
    payload_options=None,
    qf_options=None,
):
    """
    الخطوة 3: توليد الـ Stego Dataset باستخدام الموديل المُدرَّب.

    المدخلات:
        model_path    : مسار ملف الأوزان (.pth)
        data_path     : مجلد صور الـ cover
        output_folder : مجلد حفظ الصور الناتجة
        n_images      : عدد الصور المراد معالجتها
        img_size      : دقة الصور (1024 للـ IStego100K++)
        payload_options: قائمة قيم الـ payload العشوائية
        qf_options    : قائمة قيم JPEG Quality Factor
    """
    if payload_options is None:
        payload_options = [0.1, 0.2, 0.3, 0.4, 0.5]
    if qf_options is None:
        qf_options = [75, 80, 85, 90, 95]

    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── تحميل الموديل ───
    netG = Generator_Unet(3).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    netG.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    all_imgs = sorted([
        f for f in os.listdir(data_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])[:n_images]

    dataset    = StegoDataset(data_path, all_imgs, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results_log = []
    print(f"{'Image':<25} | {'PL':>4} | {'QF':>2} | {'PSNR':>8} | {'SSIM':>6}")
    print("-" * 65)

    for data in dataloader:
        img_t    = data['img'].to(device)
        img_name = data['name'][0]

        curr_pl = random.choice(payload_options)
        curr_qf = random.choice(qf_options)

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            prob  = netG(img_t)
            noise = torch.rand_like(prob)
            modi  = 0.5 * (
                torch.tanh((prob + 2 * noise - 2) * 60)
                - torch.tanh((prob - 2 * noise) * 60)
            )
            stego_t = (img_t * 255 + modi) / 255.0

        psnr_v = calculate_psnr(img_t[0], stego_t[0])
        ssim_v = calculate_ssim(img_t[0], stego_t[0])

        stego_np  = (np.clip(stego_t[0].cpu().permute(1,2,0).numpy(),0,1)*255).astype(np.uint8)
        save_name = os.path.splitext(img_name)[0] + "_stego.jpg"
        Image.fromarray(stego_np).save(
            os.path.join(output_folder, save_name),
            format='JPEG', quality=curr_qf
        )
        results_log.append([img_name, curr_pl, curr_qf,
                             round(psnr_v, 2), round(ssim_v, 4)])
        print(f"{img_name[:25]:<25} | {curr_pl:>4} | {curr_qf:>2} | "
              f"{psnr_v:>8.2f} | {ssim_v:>6.4f}")

    # ─── حفظ CSV ───
    csv_path = os.path.join(output_folder, 'dataset_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_Name','Payload','QF','PSNR','SSIM'])
        writer.writerows(results_log)

    print(f"\n✅ تم الانتهاء! الصور والتقرير في: {output_folder}")
    return results_log


# ─────────────────────────────────────────────
#  SECTION 9: MAIN.PY — Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':

    # ══════════════════════════════════════════
    #  إعدادات عامة — عدّل هذه القيم حسب بيئتك
    # ══════════════════════════════════════════
    TRAIN_MODE   = True              # True=تدريب | False=استدلال فقط
    USE_CUDA     = True
    IMG_NC       = 1                 # 1=grayscale | 3=RGB
    PAYLOAD      = 0.4               # معدل الإخفاء (0..1)
    EPOCHS       = 600
    LR           = 1e-4
    BATCH_SIZE   = 20
    BILINEAR     = False
    DATA_DIR     = '/data/BossClf/BOSSBase_256'   # ← غيّر هذا

    # مسارات لمرحلة الاستدلال
    PRETRAINED   = './models/netG_epoch_600.pth'
    EVAL_OUT     = './eval_result/'

    # مسارات لتوليد الـ dataset
    COVER_PATH   = '/path/to/istego100k/cover'   # ← غيّر هذا
    OUTPUT_PATH  = '/path/to/output_stego'        # ← غيّر هذا

    # ─── Device ───
    device = torch.device(
        "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
    )
    print(f"CUDA Available: {torch.cuda.is_available()} | Device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    if TRAIN_MODE:
        # ══ وضع التدريب ══
        dataset    = MyDataset(DATA_DIR, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)
        trainer = UTGAN(device, IMG_NC, LR, PAYLOAD, BILINEAR)
        trainer.train(dataloader, EPOCHS)

    else:
        # ══ وضع الاستدلال (Inference) ══
        dataset    = MyDataset(DATA_DIR, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=1)
        model = Generator_Unet(IMG_NC, 1, BILINEAR)
        model.load_state_dict(torch.load(PRETRAINED), strict=False)
        model.to(device).eval()

        os.makedirs(EVAL_OUT, exist_ok=True)
        for i, batch in enumerate(dataloader):
            imgs     = batch['img'].to(device)
            prob_map = model(imgs)
            imsave_single(prob_map, os.path.join(EVAL_OUT, f'{i}.tif'))

        # ══ توليد الـ Dataset (IStego100K++) ══
        # قم بتفعيل هذا عند الحاجة لتوليد الداتاسيت
        # generate_stego_dataset(
        #     model_path    = PRETRAINED,
        #     data_path     = COVER_PATH,
        #     output_folder = OUTPUT_PATH,
        #     n_images      = 200,
        #     img_size      = 1024,
        # )
