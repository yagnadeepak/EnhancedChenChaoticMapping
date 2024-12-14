import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from skimage.metrics import structural_similarity as ssim
import random

# Chen Chaotic System
def chen_chaotic_system(init, t, a, b, c, d, k):
    w, x, y, z = init
    dw = a * (x - w)
    dx = -w * y + d * w + c * x - z
    dy = w * x - b * y
    dz = w + k
    return [dw, dx, dy, dz]

# Generate chaotic matrix with Chen system
def generate_chen_matrix(size, seed):
    random.seed(seed)
    a = random.uniform(30, 50)
    b = random.uniform(2, 5)
    c = random.uniform(20, 40)
    d = random.uniform(10, 25)
    k = random.uniform(0.1, 0.5)
    initial_conditions = [random.uniform(-1, 1) for _ in range(4)]

    # Print seed and parameters for record
    print(f"Generating Chen matrix with seed: {seed}")
    print(f"Parameters: a={a}, b={b}, c={c}, d={d}, k={k}")
    print(f"Initial conditions: {initial_conditions}")

    t = np.linspace(0, 1, size * size)
    chaotic_sequence = odeint(chen_chaotic_system, initial_conditions, t, args=(a, b, c, d, k))
    w_sequence = chaotic_sequence[:, 0]
    w_matrix = np.array(w_sequence).reshape((size, size))
    w_matrix = np.uint8(255 * w_matrix / np.max(w_matrix))
    return w_matrix

# Rearrange function for each channel of an RGB image based on a chaotic key matrix
def rearrange_pixels(image, chaotic_key):
    r_channel, g_channel, b_channel = cv2.split(image)
    permutation = np.argsort(chaotic_key.flatten())

    rearranged_r = np.zeros_like(r_channel.flatten())
    rearranged_g = np.zeros_like(g_channel.flatten())
    rearranged_b = np.zeros_like(b_channel.flatten())

    rearranged_r[permutation] = r_channel.flatten()
    rearranged_g[permutation] = g_channel.flatten()
    rearranged_b[permutation] = b_channel.flatten()

    return cv2.merge([
        rearranged_r.reshape(r_channel.shape),
        rearranged_g.reshape(g_channel.shape),
        rearranged_b.reshape(b_channel.shape)
    ])

# XOR operation with key matrix for each RGB channel
def xor_with_key(image, key):
    r_channel, g_channel, b_channel = cv2.split(image)
    r_encrypted = np.bitwise_xor(r_channel, key)
    g_encrypted = np.bitwise_xor(g_channel, key)
    b_encrypted = np.bitwise_xor(b_channel, key)
    return cv2.merge([r_encrypted, g_encrypted, b_encrypted])

# Calculate PSNR
def calculate_psnr(original, encrypted):
    # Ensure images are of type float for precise calculations
    original = original.astype(np.float64)
    encrypted = encrypted.astype(np.float64)

    mse = np.mean((original - encrypted) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / mse)

# Calculate SSIM
def calculate_ssim(original, encrypted):
    return ssim(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(encrypted, cv2.COLOR_BGR2GRAY))

# Calculate NPCR and UACI
def calculate_npcr_uaci(original, encrypted):
    original = original.astype(np.float64)
    encrypted = encrypted.astype(np.float64)

    # Normalize images if needed
    diff = (original != encrypted).astype(np.float64)
    npcr = (np.sum(diff) / original.size) * 100

    # Calculate UACI ensuring correct scaling
    uaci = (np.mean(np.abs(original - encrypted) / 255) * 100)

    return npcr, uaci

# Calculate correlation coefficients for different directions
def calculate_correlation(image, encrypted):
    directions = ["horizontal", "vertical", "diagonal"]
    correlation = {}
    for direction in directions:
        if direction == "horizontal":
            correlation[direction] = np.corrcoef(image[:, :-1].flatten(), encrypted[:, 1:].flatten())[0, 1]
        elif direction == "vertical":
            correlation[direction] = np.corrcoef(image[:-1, :].flatten(), encrypted[1:, :].flatten())[0, 1]
        elif direction == "diagonal":
            correlation[direction] = np.corrcoef(image[:-1, :-1].flatten(), encrypted[1:, 1:].flatten())[0, 1]
    return correlation

# Plot correlation between original and encrypted images
def plot_correlation(image, encrypted):
    directions = ["horizontal", "vertical", "diagonal"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, direction in enumerate(directions):
        if direction == "horizontal":
            original_pixels = image[:, :-1].flatten()
            encrypted_pixels = encrypted[:, 1:].flatten()
        elif direction == "vertical":
            original_pixels = image[:-1, :].flatten()
            encrypted_pixels = encrypted[1:, :].flatten()
        elif direction == "diagonal":
            original_pixels = image[:-1, :-1].flatten()
            encrypted_pixels = encrypted[1:, 1:].flatten()

        axs[i].scatter(original_pixels, encrypted_pixels, s=1, alpha=0.5)
        axs[i].set_title(f"{direction.capitalize()} Correlation")
        axs[i].set_xlabel("Original Image Pixels")
        axs[i].set_ylabel("Encrypted Image Pixels")
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

def plot_correlation(image, encrypted):
    directions = ["horizontal", "vertical", "diagonal"]
    correlation_values = calculate_correlation(image, encrypted)  # Calculate correlation for each direction
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, direction in enumerate(directions):
        if direction == "horizontal":
            original_pixels = image[:, :-1].flatten()
            encrypted_pixels = encrypted[:, 1:].flatten()
        elif direction == "vertical":
            original_pixels = image[:-1, :].flatten()
            encrypted_pixels = encrypted[1:, :].flatten()
        elif direction == "diagonal":
            original_pixels = image[:-1, :-1].flatten()
            encrypted_pixels = encrypted[1:, 1:].flatten()

        axs[i].scatter(original_pixels, encrypted_pixels, s=1, alpha=0.5)
        axs[i].set_title(f"{direction.capitalize()} Correlation")
        axs[i].set_xlabel("Original Image Pixels")
        axs[i].set_ylabel("Encrypted Image Pixels")
        axs[i].grid(True)

        # Display the correlation value on the plot
        corr_value = correlation_values[direction]
        axs[i].text(0.05, 0.95, f"Correlation: {corr_value:.4f}", transform=axs[i].transAxes,
                    fontsize=12, verticalalignment='top', color='red')

    plt.tight_layout()
    plt.show()

# Multi-layer encryption with additional rounds to reduce PSNR and UACI values
def encrypt_image_with_layers(original_image, chaotic_key1, chaotic_key2, grayscale_image1, grayscale_image2, rounds=5):
    encrypted_image = original_image.copy()
    matrix_size = chaotic_key1.shape[0]

    for _ in range(rounds):
        # Generate new keys for each round
        new_seed1, new_seed2 = random.randint(0, 10000), random.randint(0, 10000)
        round_key1 = generate_chen_matrix(matrix_size, new_seed1)
        round_key2 = generate_chen_matrix(matrix_size, new_seed2)

        # Add a small noise component to disrupt the pixel arrangement
        noise = np.random.randint(0, 10, (matrix_size, matrix_size), dtype=np.uint8)

        # First round of rearrangement and XORing with added noise
        rearranged_image = rearrange_pixels(encrypted_image, round_key1 + noise)
        encrypted_image = xor_with_key(rearranged_image, round_key1)
        encrypted_image = xor_with_key(encrypted_image, grayscale_image1)

        # Second round of rearrangement and XORing with different keys
        rearranged_image = rearrange_pixels(encrypted_image, round_key2 + noise)
        encrypted_image = xor_with_key(rearranged_image, round_key2)
        encrypted_image = xor_with_key(encrypted_image, grayscale_image2)

    return encrypted_image

# Main function to execute the process
def main(image_paths, grayscale_path1, grayscale_path2):
    matrix_size = 256

    for idx, image_path in enumerate(image_paths):
        seed1, seed2 = random.randint(0, 10000), random.randint(0, 10000)
        chaotic_key1 = generate_chen_matrix(matrix_size, seed1)
        chaotic_key2 = generate_chen_matrix(matrix_size, seed2)

        grayscale_image1 = cv2.resize(cv2.imread(grayscale_path1, cv2.IMREAD_GRAYSCALE), (matrix_size, matrix_size))
        grayscale_image2 = cv2.resize(cv2.imread(grayscale_path2, cv2.IMREAD_GRAYSCALE), (matrix_size, matrix_size))

        original_image = cv2.resize(cv2.imread(image_path), (matrix_size, matrix_size))

        # Perform encryption with more rounds and dynamic keys for stronger disruption
        encrypted_image = encrypt_image_with_layers(original_image, chaotic_key1, chaotic_key2, grayscale_image1, grayscale_image2, rounds=5)

        psnr_value = calculate_psnr(original_image, encrypted_image)
        ssim_value = calculate_ssim(original_image, encrypted_image)
        npcr, uaci = calculate_npcr_uaci(original_image, encrypted_image)
        correlation_values = calculate_correlation(original_image, encrypted_image)

        print(f"\nImage {idx + 1} - Performance Metrics")
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
        print(f"NPCR: {npcr:.2f}%")
        print(f"UACI: {uaci:.2f}%")
        print("Correlation values:", correlation_values)

        colors = ('r', 'g', 'b')
        plt.figure(figsize=(12, 16))

        plt.subplot(4, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(4, 2, 2)
        plt.imshow(cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB))
        plt.title("Encrypted Image")
        plt.axis('off')

        for i, color in enumerate(colors):
            original_hist = cv2.calcHist([original_image], [i], None, [256], [0, 256])
            plt.subplot(4, 2, 3 + i)
            plt.fill_between(range(256), original_hist.flatten(), color=color, alpha=0.6)
            plt.title(f"Original {color.upper()} Histogram")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")

        for i, color in enumerate(colors):
            encrypted_hist = cv2.calcHist([encrypted_image], [i], None, [256], [0, 256])
            plt.subplot(4, 2, 6 + i)
            plt.fill_between(range(256), encrypted_hist.flatten(), color=color, alpha=0.6)
            plt.title(f"Encrypted {color.upper()} Histogram")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

        # Plot correlation in all formats
        plot_correlation(original_image, encrypted_image)

# Example usage remains the same
image_paths = ["UPLOAD IMAGE PATHS"]

grayscale_path1 = "UPLOAD GREYSCALE IMAGE PATHS"
grayscale_path2 = "UPLOAD GREYSCALE IMAGE PATHS"
main(image_paths, grayscale_path1, grayscale_path2)