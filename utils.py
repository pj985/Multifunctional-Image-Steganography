import numpy as np
from PIL import Image
import torch.nn as nn
from math import log10
import torch, cv2, os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def plot_histograms(cover_images, stega_images, save_path, num_bins=10):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Histograms: Cover vs Stego')
    
    for i in range(3):
        cover_hist, bin_edges = np.histogram(cover_images[0, i].cpu().numpy().flatten(), bins=num_bins, range=(0, 1))
        stego_hist, _ = np.histogram(stega_images[0, i].cpu().numpy().flatten(), bins=num_bins, range=(0, 1))
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        axs[0, i].bar(bin_centers, cover_hist, width=1/num_bins, color='blue', alpha=0.7)
        axs[0, i].set_title(f'Cover Image - Channel {i}')
        axs[0, i].set_xlabel('Pixel Value')
        axs[0, i].set_ylabel('Frequency')
        
        axs[1, i].bar(bin_centers, stego_hist, width=1/num_bins, color='red', alpha=0.7)
        axs[1, i].set_title(f'Stego Image - Channel {i}')
        axs[1, i].set_xlabel('Pixel Value')
        axs[1, i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(secret_images, outputs, cover_images, stega_images, filenames, args):
    results = []
    for i, (img_orig, img_recon, img_cover, img_stego, fname) in enumerate(zip(
        secret_images, outputs, cover_images, stega_images, filenames)):
        
        secret_psnr = calc_psnr((img_recon.permute(1, 2, 0)*255).cpu().numpy(),
                                (img_orig.permute(1, 2, 0)*255).cpu().numpy())
        secret_ssim = calc_ssim((img_recon.permute(1, 2, 0)*255).cpu().numpy(),
                                (img_orig.permute(1, 2, 0)*255).cpu().numpy())
        
        stego_psnr = calc_psnr((img_stego.permute(1, 2, 0)*255).cpu().numpy(),
                               (img_cover.permute(1, 2, 0)*255).cpu().numpy())
        stego_ssim = calc_ssim((img_stego.permute(1, 2, 0)*255).cpu().numpy(),
                               (img_cover.permute(1, 2, 0)*255).cpu().numpy())
        
        results.append({
            'filename': fname,
            'secret_psnr': secret_psnr,
            'secret_ssim': secret_ssim,
            'stego_psnr': stego_psnr,
            'stego_ssim': stego_ssim
        })
        
        print(f'Image {fname}:')
        print(f'Secret - PSNR = {secret_psnr:.2f}, SSIM = {secret_ssim:.4f}')
        print(f'Stego - PSNR = {stego_psnr:.2f}, SSIM = {stego_ssim:.4f}')
        
        with open(args.log_dir, "a") as f:
            f.write(f"File: {fname}\n")
            f.write(f"Secret - PSNR: {secret_psnr:.2f}\tSSIM: {secret_ssim:.4f}\n")
            f.write(f"Stego - PSNR: {stego_psnr:.2f}\tSSIM: {stego_ssim:.4f}\n\n")
    
    return results


def plot_histograms(cover_images, stega_images, save_path, num_bins=10):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Histograms: Cover vs Stego')
    
    for i in range(3):
        cover_hist, bin_edges = np.histogram(cover_images[0, i].cpu().numpy().flatten(), bins=num_bins, range=(0, 1))
        stego_hist, _ = np.histogram(stega_images[0, i].cpu().numpy().flatten(), bins=num_bins, range=(0, 1))
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        axs[0, i].bar(bin_centers, cover_hist, width=1/num_bins, color='blue', alpha=0.7)
        axs[0, i].set_title(f'Cover Image - Channel {i}')
        axs[0, i].set_xlabel('Pixel Value')
        axs[0, i].set_ylabel('Frequency')
        
        axs[1, i].bar(bin_centers, stego_hist, width=1/num_bins, color='red', alpha=0.7)
        axs[1, i].set_title(f'Stego Image - Channel {i}')
        axs[1, i].set_xlabel('Pixel Value')
        axs[1, i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class HighPassFilter(nn.Module):
    def __init__(self):
        super(HighPassFilter, self).__init__()
        self.requires_grad = True
        high_pass_kernel = torch.tensor([[
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]], dtype=torch.float32)
        self.kernel = high_pass_kernel.unsqueeze(0).repeat(3, 1, 1, 1)
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)
        self.conv.weight = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        return self.conv(x)


class MultiScaleMSE(nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.weights = weights

    def forward(self, pred, target):
        loss = 0.0
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1:
                resized_pred = F.interpolate(pred, scale_factor=scale, mode='bilinear')
                resized_target = F.interpolate(target, scale_factor=scale, mode='bilinear')
            else:
                resized_pred, resized_target = pred, target

            loss += weight * F.mse_loss(resized_pred, resized_target)
        return loss


class MultiScalePatchMSE(nn.Module):
    def __init__(self, scales=[1, 2, 4], patch_sizes=[64, 16, 16], weights=[0.7, 0.3, 0.0]):
        super().__init__()
        self.scales = scales
        self.patch_sizes = patch_sizes
        self.weights = weights

    def extract_patches(self, x, patch_size):
        # 非重叠分块：x.shape = [B, C, H, W]
        B, C, H, W = x.shape
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(B, C, -1, patch_size, patch_size)             # [B, C, num_patches, p, p]
        return x

    def forward(self, pred, target):
        total_loss = 0.0
        for scale, patch_size, weight in zip(self.scales, self.patch_sizes, self.weights):
            # 下采样：使用双线性插值替代平均池化
            if scale > 1:
                # 计算目标尺寸（H和W均缩小scale倍）
                down_size = (pred.shape[2] // scale, pred.shape[3] // scale)
                down_pred = F.interpolate(pred, size=down_size, mode='bilinear', align_corners=False)
                down_target = F.interpolate(target, size=down_size, mode='bilinear', align_corners=False)
            else:
                down_pred, down_target = pred, target

            # 分块
            pred_patches = self.extract_patches(down_pred, patch_size)        # [B, C, num_patches, p, p]
            target_patches = self.extract_patches(down_target, patch_size)

            # 计算每个块的MSE并平均
            mse = (pred_patches - target_patches).pow(2).mean(dim=(3, 4))     # [B, C, num_patches]
            loss = weight * mse.mean()
            total_loss += loss

        return total_loss


def analyze_image(image_tensor):
    # 分析图像的基本统计信息
    print(f"Shape: {image_tensor.shape}")
    print(f"Min value: {image_tensor.min().item():.4f}")
    print(f"Max value: {image_tensor.max().item():.4f}")
    print(f"Mean value: {image_tensor.mean().item():.4f}")
    print(f"Std value: {image_tensor.std().item():.4f}")


class ImagePairDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if os.path.isfile(os.path.join(image_dir, f))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, 2*image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        img_split = torch.split(img_tensor, img_tensor.size(-1)//2, dim=2)
        return {
            'secret': img_split[0],
            'cover': img_split[1],
            'filename': os.path.basename(img_path)
        }


def shuffle_params(m):
    if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())
        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))


def calc_psnr(img1, img2):
    diff = (img1 - img2) / 255.0
    diff[:, :, 0] = diff[:, :, 0] * 65.738 / 256.0
    diff[:, :, 1] = diff[:, :, 1] * 129.057 / 256.0
    diff[:, :, 2] = diff[:, :, 2] * 25.064 / 256.0
    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * log10(mse)


def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')

class ScaleLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x + 1) / 2


import numpy as np
import cv2
from skimage.color import rgb2gray
from scipy.stats import entropy

# ===========================
# 直方图相似度、熵差、相关系数
# ===========================

def calc_hist_similarity(img1, img2, bins=64):
    """
    使用 Bhattacharyya 系数 计算灰度直方图相似度
    返回 [0,1]，越接近 1 越相似
    """
    img1 = np.array(img1.convert("L")).astype(np.float32) / 255.0
    img2 = np.array(img2.convert("L")).astype(np.float32) / 255.0

    h1, _ = np.histogram(img1.flatten(), bins=bins, range=(0,1), density=True)
    h2, _ = np.histogram(img2.flatten(), bins=bins, range=(0,1), density=True)
    h1 /= (h1.sum() + 1e-8)
    h2 /= (h2.sum() + 1e-8)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(h1 * h2))
    return float(bc)  # 典型范围：0.5~1.0

def calc_entropy(img):
    """
    计算单张图像的 Shannon 熵（单位：bit）
    """
    img = np.array(img.convert("L")).astype(np.float32) / 255.0
    hist, _ = np.histogram(img, bins=256, range=(0,1), density=True)
    hist += 1e-8
    return float(entropy(hist, base=2))  # 通常类噪声≈7.5~8.0

def calc_entropy_diff(img1, img2):
    """
    两张图像的熵差，范围 [0,1.5]
    """
    e1 = calc_entropy(img1)
    e2 = calc_entropy(img2)
    return abs(e1 - e2)

def calc_corrcoef(img1, img2):
    """
    灰度图的像素相关系数 (归一化后)
    """
    g1 = rgb2gray(np.array(img1).astype(np.float32) / 255.0).flatten()
    g2 = rgb2gray(np.array(img2).astype(np.float32) / 255.0).flatten()
    return float(np.corrcoef(g1, g2)[0, 1])  # 典型范围：0.4~0.9