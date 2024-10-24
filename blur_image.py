import cv2
import numpy as np

def apply_gaus_blur(image, blur_ksize=3):
    """应用高斯模糊"""
    return cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

def apply_frost(image):
    """应用霜冻效果"""
    frost_layer = np.random.randint(150, 255, image.shape, dtype=np.uint8)
    return cv2.addWeighted(image, 0.5, frost_layer, 0.5, 0)

def apply_gaussian_noise(image, mean=0, sigma=25):
    """应用高斯噪声"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian_noise)
    return np.clip(noisy_image, 0, 255)

def apply_glass_blur(image, blur_strength=5):
    """应用玻璃模糊效果"""
    return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

def apply_impulse_noise(image, probability=0.05):
    """应用脉冲噪声"""
    noisy_image = np.copy(image)
    random_values = np.random.rand(*image.shape[:2])
    noisy_pixels = random_values < probability
    noisy_image[noisy_pixels] = 255  # 白色噪声
    return noisy_image

def apply_jpeg_compression(image, quality=10):
    """应用JPEG压缩"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_motion_blur(image, kernel_size=5):
    """应用运动模糊"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_pixelate(image, pixel_size=4):
    """应用像素化效果"""
    h, w = image.shape[:2]
    image_small = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(image_small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_saturate(image, factor=1.5):
    """应用饱和度"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_shot_noise(image, probability=0.01):
    """应用散点噪声"""
    noisy_image = np.copy(image)
    num_noise = int(probability * image.size / 3)
    coords = [np.random.randint(0, i - 1, num_noise) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255  # 设置为白色
    return noisy_image

def apply_snow(image):
    """应用雪花效果"""
    snow_layer = np.random.randint(0, 255, image.shape, dtype=np.uint8)
    return cv2.addWeighted(image, 0.7, snow_layer, 0.3, 0)

def apply_spatter(image):
    """应用喷溅效果"""
    spatter_layer = np.random.randint(0, 255, image.shape, dtype=np.uint8)
    return cv2.addWeighted(image, 0.5, spatter_layer, 0.5, 0)

def apply_speckle_noise(image):
    """应用斑点噪声"""
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255)

def apply_zoom_blur(image, zoom_strength=3):
    """应用缩放模糊"""
    return cv2.GaussianBlur(image, (zoom_strength, zoom_strength), 0)
