import numpy as np
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# --------------------------
# PredictiveCoding2D class
# --------------------------
class PredictiveCoding2D:
    def __init__(self, num_bits=2):
        self.num_bits = num_bits
        self.quantization_params = None

    # --------------------------
    # Load image
    # --------------------------
    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path does not exist: {path}")
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return np.array(img, dtype=np.float32)

    # --------------------------
    # JPEG-LS Predictor
    # --------------------------
    def predictiveImageGenerate(self, img):
        self.original = img.astype(np.float32)
        h, w, c = self.original.shape
        self.predicted = np.zeros_like(self.original)

        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    A = self.original[y, x-1, ch] if x > 0 else 0
                    B = self.original[y-1, x, ch] if y > 0 else 0
                    C = self.original[y-1, x-1, ch] if (x > 0 and y > 0) else 0

                    # JPEG-LS predictor
                    if C >= max(A, B):
                        P = min(A, B)
                    elif C <= min(A, B):
                        P = max(A, B)
                    else:
                        P = A + B - C

                    self.predicted[y, x, ch] = P

        self.error = self.original - self.predicted
        return self.predicted, self.error

    # --------------------------
    # Uniform quantization - SEPARATE PER CHANNEL
    # --------------------------
    def quantize_dequantize(self, error):
        levels = 2 ** self.num_bits
        h, w, c = error.shape

        quantized = np.zeros_like(error, dtype=np.int32)
        dequantized = np.zeros_like(error, dtype=np.float32)

        # Store quantization parameters for each channel
        self.quantization_params = {
            'e_min': np.zeros(c, dtype=np.float32),
            'e_max': np.zeros(c, dtype=np.float32),
            'step': np.zeros(c, dtype=np.float32),
            'levels': levels,
            'num_bits': self.num_bits
        }

        # Process each channel separately
        for ch in range(c):
            error_ch = error[:, :, ch]
            e_min, e_max = error_ch.min(), error_ch.max()
            step = (e_max - e_min) / levels
            if step == 0:
                step = 1e-6

            # Quantize this channel
            quantized_ch = np.floor((error_ch - e_min) / step).astype(np.int32)
            quantized_ch = np.clip(quantized_ch, 0, levels - 1)

            # Dequantize this channel
            dequantized_ch = e_min + (quantized_ch + 0.5) * step

            # Store results
            quantized[:, :, ch] = quantized_ch
            dequantized[:, :, ch] = dequantized_ch

            # Store parameters for this channel
            self.quantization_params['e_min'][ch] = e_min
            self.quantization_params['e_max'][ch] = e_max
            self.quantization_params['step'][ch] = step

        return quantized, dequantized

    # --------------------------
    # Save image helper
    # --------------------------
    def save_vis(self, img, path):
        img_clipped = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img_clipped).save(path)

    # --------------------------
    # Save quantized as image (visualization)
    # --------------------------
    def save_quantized_image(self, quantized, output_path):
        """Save quantized data as a visualizable image"""
        # Scale quantized values to 0-255 for visualization
        # Process each channel separately for better visualization
        quantized_img = np.zeros_like(quantized, dtype=np.uint8)
        for ch in range(3):
            quantized_ch = quantized[:, :, ch]
            if quantized_ch.max() > quantized_ch.min():
                quantized_img[:, :, ch] = ((quantized_ch - quantized_ch.min()) /
                                         (quantized_ch.max() - quantized_ch.min()) * 255).astype(np.uint8)
        Image.fromarray(quantized_img).save(output_path)
        print(f"Quantized image saved to: {output_path}")
        return quantized_img

    # --------------------------
    # Save quantized as npz (binary data)
    # --------------------------
    def save_quantized_npz(self, quantized, quant_params, original_image, output_path):
        """Save quantized data and parameters as .npz file"""
        h, w, c = quantized.shape

        # Store the quantized data EXCLUDING first row and first column
        quantized_core = quantized[1:, 1:, :].copy()  # Shape: (h-1, w-1, c)

        # Store original first row and first column
        original_first_row = original_image[0, :, :].astype(np.float32)
        original_first_col = original_image[:, 0, :].astype(np.float32)

        np.savez_compressed(
            output_path,
            quantized_core=quantized_core,  # Core quantized data (no first row/col)
            e_min=quant_params['e_min'],  # Array of 3 values, one per channel
            e_max=quant_params['e_max'],  # Array of 3 values, one per channel
            step=quant_params['step'],    # Array of 3 values, one per channel
            levels=quant_params['levels'],
            num_bits=quant_params['num_bits'],
            original_first_row=original_first_row,
            original_first_col=original_first_col,
            original_shape=original_image.shape
        )
        print(f"Quantized data (npz) saved to: {output_path}")
        print(f"  - Quantized core shape (no first row/col): {quantized_core.shape}")
        print(f"  - Original first row/col stored separately")
        print(f"  - Quantization params per channel:")
        for ch in range(c):
            print(f"    Channel {ch}: e_min={quant_params['e_min'][ch]:.2f}, "
                  f"e_max={quant_params['e_max'][ch]:.2f}, step={quant_params['step'][ch]:.2f}")

    # --------------------------
    # Load quantized from npz
    # --------------------------
    def load_quantized_npz(self, quantized_path):
        """Load quantized data from .npz file"""
        data = np.load(quantized_path)

        quant_params = {
            'e_min': data['e_min'],
            'e_max': data['e_max'],
            'step': data['step'],
            'levels': data['levels'].item(),
            'num_bits': data['num_bits'].item(),
            'original_first_row': data['original_first_row'],
            'original_first_col': data['original_first_col'],
            'original_shape': tuple(data['original_shape']),
            'quantized_core': data['quantized_core']
        }

        return data['quantized_core'], quant_params

    # --------------------------
    # Dequantize - SEPARATE PER CHANNEL
    # --------------------------
    def dequantize(self, quantized, quant_params):
        """Dequantize using saved parameters - separate per channel"""
        e_min = quant_params['e_min']  # Array of 3 values
        step = quant_params['step']    # Array of 3 values
        c = quantized.shape[2] if len(quantized.shape) > 2 else 1

        if c == 1:
            # Single channel
            dequantized = e_min + (quantized + 0.5) * step
        else:
            # Multiple channels - process each separately
            dequantized = np.zeros_like(quantized, dtype=np.float32)
            for ch in range(c):
                dequantized[:, :, ch] = e_min[ch] + (quantized[:, :, ch] + 0.5) * step[ch]

        return dequantized

    # --------------------------
    # Generate predicted image from first row and first column
    # --------------------------
    def generate_predicted_from_border(self, first_row, first_col, h, w, c):
        """Generate predicted image using only first row and first column"""
        predicted = np.zeros((h, w, c), dtype=np.float32)

        # Set first row and first column
        predicted[0, :, :] = first_row
        predicted[:, 0, :] = first_col

        # Generate predicted values for the rest
        for y in range(1, h):
            for x in range(1, w):
                for ch in range(c):
                    A = predicted[y, x-1, ch]  # Left
                    B = predicted[y-1, x, ch]  # Top
                    C = predicted[y-1, x-1, ch]  # Top-left

                    # JPEG-LS predictor
                    if C >= max(A, B):
                        P = min(A, B)
                    elif C <= min(A, B):
                        P = max(A, B)
                    else:
                        P = A + B - C

                    predicted[y, x, ch] = P

        return predicted

    # --------------------------
    # Decompress SOLELY from quantized npz file - SIMPLE VERSION
    # --------------------------
    def decompress_from_npz(self, quantized_npz_path, output_name="decompressed_from_npz.png"):
        """Decompress image SOLELY from quantized.npz file"""

        # Check if file exists
        if not os.path.exists(quantized_npz_path):
            print(f"Error: File not found: {quantized_npz_path}")
            default_path = os.path.join("output", "4_quantized.npz")
            if os.path.exists(default_path):
                print(f"Trying default path: {default_path}")
                quantized_npz_path = default_path
            else:
                return None

        # Load quantized data and parameters
        quantized_core, quant_params = self.load_quantized_npz(quantized_npz_path)

        # Get original shape
        h, w, c = quant_params['original_shape']

        # Get stored original first row and first column
        original_first_row = quant_params['original_first_row']
        original_first_col = quant_params['original_first_col']

        print(f"Starting decompression from npz...")
        print(f"Image shape: {h}x{w}x{c}")
        print(f"Quantized core shape: {quantized_core.shape}")

        # STEP 1: Dequantize the error core - each channel separately
        dequantized_core = self.dequantize(quantized_core, quant_params)

        # Create full dequantized error array with zeros for first row/col
        dequantized_full = np.zeros((h, w, c), dtype=np.float32)
        dequantized_full[1:, 1:, :] = dequantized_core

        print(f"\nDequantized error statistics (per channel):")
        for ch in range(c):
            print(f"  Channel {ch}: Core min={dequantized_core[:,:,ch].min():.2f}, "
                  f"max={dequantized_core[:,:,ch].max():.2f}")

        # STEP 2: Generate predicted image from first row and first column
        print(f"\nGenerating predicted image from border...")
        predicted = self.generate_predicted_from_border(original_first_row, original_first_col, h, w, c)

        print(f"Predicted image statistics:")
        print(f"  Min: {predicted.min():.2f}, Max: {predicted.max():.2f}")

        # STEP 3: Add predicted + dequantized error
        reconstructed = predicted + dequantized_full

        print(f"\nReconstructed image statistics:")
        print(f"  Before clipping - Min: {reconstructed.min():.2f}, Max: {reconstructed.max():.2f}")

        # Clip to valid range
        reconstructed = np.clip(reconstructed, 0, 255)

        print(f"  After clipping - Min: {reconstructed.min():.2f}, Max: {reconstructed.max():.2f}")
        print(f"  Mean: {reconstructed.mean():.2f}")

        # Save the reconstructed image
        folder = "output"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, output_name)

        # Convert to uint8 and save
        reconstructed_uint8 = reconstructed.astype(np.uint8)
        Image.fromarray(reconstructed_uint8).save(output_path)

        print(f"\n" + "="*60)
        print(f"DECOMPRESSION FROM NPZ COMPLETE!")
        print(f"Output saved as: {output_path}")
        print(f"Reconstructed solely from quantized.npz file")
        print("="*60)

        return reconstructed

    # --------------------------
    # Compression Ratio
    # --------------------------
    def calculate_compression_ratio(self, original_image, num_bits):
        if not isinstance(original_image, np.ndarray):
            if hasattr(self, 'original'):
                original_image = self.original
            else:
                return None, "Error: Original image data not found."

        original_bpp = 8.0 * 3  # 8 bits per channel, 3 channels
        compressed_bpp = float(num_bits) * 3  # num_bits per channel, 3 channels

        compression_ratio = original_bpp / compressed_bpp

        # Original File Size
        original_size_bytes = original_image.size * (8.0 / 8.0)  # Already in bytes for 8-bit per channel

        # Compressed Data Size
        compressed_size_bytes = original_image.size * (num_bits / 8.0)  # Convert bits to bytes

        ratio_string = f"{round(compression_ratio, 2)}:1"

        return ratio_string, original_size_bytes, compressed_size_bytes

    # --------------------------
    # Compress - Saves both quantized image and npz
    # --------------------------
    def compress(self, img_path):
        original = self.load_image(img_path)
        predicted, error = self.predictiveImageGenerate(original)
        quantized, dequantized = self.quantize_dequantize(error)

        folder = "output"
        os.makedirs(folder, exist_ok=True)

        # Save all images
        self.save_vis(original, os.path.join(folder, "1_original.png"))
        self.save_vis(predicted, os.path.join(folder, "2_predicted.png"))
        self.save_vis(error + 128, os.path.join(folder, "3_error.png"))

        # Save quantized as IMAGE (visualization)
        quantized_img_path = os.path.join(folder, "4_quantized.png")
        self.save_quantized_image(quantized, quantized_img_path)

        # Save quantized as NPZ (binary data for decompression)
        quantized_npz_path = os.path.join(folder, "4_quantized.npz")
        self.save_quantized_npz(quantized, self.quantization_params, original, quantized_npz_path)

        # Save dequantized
        dequantized_path = os.path.join(folder, "5_dequantized.png")
        self.save_vis(dequantized, dequantized_path)

        # Create and save decompressed image from images
        self.decompress("./output/2_predicted.png", "./output/5_dequantized.png", "6_decompressed.png")

        # --- Calculate and Display Compression Ratio ---
        ratio_str, original_size, compressed_size = self.calculate_compression_ratio(original, self.num_bits)
        if ratio_str:
            print("\n" + "=" * 60)
            print(f"Compression Ratio ({8*3} bits / {self.num_bits*3} bits): {ratio_str}")
            print(f"Original size (estimated): {round(original_size / 1024, 2)} KB")
            print(f"Compressed data size (estimated): {round(compressed_size / 1024, 2)} KB")
            print("=" * 60)

        print("\n" + "="*60)
        print("COMPRESSION COMPLETE! Files saved in 'output' folder:")
        print("="*60)
        print("1. Original image:              1_original.png")
        print("2. Predicted image:             2_predicted.png")
        print("3. Error (with +128 offset):    3_error.png")
        print("4. Quantized (IMAGE version):   4_quantized.png")
        print("5. Quantized (NPZ version):     4_quantized.npz")
        print("   - Core quantized data (no first row/col)")
        print("   - Original first row/col stored separately")
        print("   - Separate quantization params per channel")
        print("   - Can decompress SOLELY from this file")
        print("6. Dequantized error:           5_dequantized.png")
        print("7. Final decompressed:          6_decompressed.png")
        print("="*60)

    # --------------------------
    # Decompress from images (original method)
    # --------------------------
    def decompress(self, predicted_path, dequantized_path, output_name="decompressed.png"):
        """Decompress using predicted and dequantized images"""
        predicted = self.load_image(predicted_path)
        dequantized = self.load_image(dequantized_path)

        # The dequantized image is the actual error values (not offset)
        # So we can directly add them
        decompressed = predicted + dequantized

        folder = "output"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, output_name)
        self.save_vis(decompressed, output_path)

        print(f"Decompressed image saved as: {output_path}")
        return decompressed

    # --------------------------
    # Print Quantization Table - SEPARATE PER CHANNEL
    # --------------------------
    def print_quantization_table(self):
        if not self.quantization_params:
            print("Quantization parameters not available. Run compression first.")
            return

        params = self.quantization_params
        levels = params['levels']
        N = params['num_bits']
        e_min = params['e_min']
        step = params['step']

        num_channels = len(e_min) if hasattr(e_min, '__len__') else 1

        for ch in range(num_channels):
            e_min_ch = e_min[ch] if hasattr(e_min, '__len__') else e_min
            step_ch = step[ch] if hasattr(step, '__len__') else step

            header_width = 110

            print("\n" + "=" * header_width)
            print(f"QUANTIZATION TABLE - CHANNEL {ch} ({N} BITS, {levels} LEVELS)")
            print("=" * header_width)
            print(
                f"Error Min (e_min): {e_min_ch:.4f} | Error Max (e_max): {params['e_max'][ch]:.4f} | Step Size: {step_ch:.4f}")
            print("-" * header_width)

            # Updated Header: Changed 'Input Error Range [Lower, Upper)'
            print(
                f"| {'Index':<7} | {'Binary Code':<15} | {'De-quantized Value':<25} | {'Input Error Range Lower, Upper':<45} |")
            print("-" * header_width)

            for i in range(levels):
                index = i
                binary_code = format(i, f'0{N}b')

                lower_bound = e_min_ch + i * step_ch
                upper_bound = e_min_ch + (i + 1) * step_ch

                # Change 1: Updated range string format
                range_str = f"{lower_bound:.4f}, {upper_bound:.4f}"

                dequant_value = e_min_ch + (i + 0.5) * step_ch

                print(f"| {index:<7} | {binary_code:<15} | {dequant_value:.8f}{'':<17} | {range_str:<45} |")

            print("=" * header_width)


# --------------------------
# CLI class with default values
# --------------------------
class CLI:
    def run(self):
        while True:
            print("\n" + "="*50)
            print("PREDICTIVE CODING COMPRESSION SYSTEM")
            print("="*50)
            print("1. Compress image (saves quantized as both PNG and NPZ)")
            print("2. Decompress SOLELY from quantized NPZ file")
            print("3. Decompress from predicted + dequantized images")
            print("4. Exit")
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == "4":
                print("Exiting...")
                return

            if choice == "1":
                # Default image path suggestion
                default_img = "input.jpg"  # or "input.png"
                img_path = input(f"Enter image file path (jpg/png) [default: {default_img}]: ").strip()
                if not img_path:
                    img_path = default_img

                if not os.path.exists(img_path):
                    print(f"Error: File '{img_path}' not found!")
                    continue

                try:
                    num_bits = int(input("Enter number of bits for quantization (default=2): ").strip() or 2)
                    if num_bits < 1 or num_bits > 8:
                        print("Warning: Using 2 bits (recommended 1-8 bits)")
                        num_bits = 2
                except ValueError:
                    num_bits = 2

                pc = PredictiveCoding2D(num_bits=num_bits)
                pc.compress(img_path)
                pc.print_quantization_table()

            elif choice == "2":
                print("\n--- Decompress SOLELY from NPZ file ---")
                print("This will reconstruct the image using ONLY the quantized.npz file")
                print("No predicted image needed!")

                # Default paths
                default_npz = os.path.join("output", "4_quantized.npz")
                quantized_npz_path = input(f"Enter quantized NPZ file path [default: {default_npz}]: ").strip()
                if not quantized_npz_path:
                    quantized_npz_path = default_npz

                if not os.path.exists(quantized_npz_path):
                    print(f"Error: File '{quantized_npz_path}' not found!")
                    print("Please run compression first (option 1) to create the .npz file.")
                    continue

                output_name = input("Enter output filename (default=decompressed_from_npz.png): ").strip()
                output_name = output_name if output_name else "decompressed_from_npz.png"

                pc = PredictiveCoding2D()
                result = pc.decompress_from_npz(quantized_npz_path, output_name)
                if result is None:
                    print("Decompression failed!")

            elif choice == "3":
                print("\n--- Decompress from images ---")

                # Default paths
                default_predicted = os.path.join("output", "2_predicted.png")
                default_dequantized = os.path.join("output", "5_dequantized.png")

                predicted_path = input(f"Enter predicted image path [default: {default_predicted}]: ").strip()
                if not predicted_path:
                    predicted_path = default_predicted

                dequantized_path = input(f"Enter dequantized image path [default: {default_dequantized}]: ").strip()
                if not dequantized_path:
                    dequantized_path = default_dequantized

                if not os.path.exists(predicted_path) or not os.path.exists(dequantized_path):
                    print("Error: One or both files not found!")
                    print("Please run compression first (option 1) to create these files.")
                    continue

                output_name = input("Enter output filename (default=decompressed.png): ").strip()
                output_name = output_name if output_name else "decompressed.png"

                pc = PredictiveCoding2D()
                pc.decompress(predicted_path, dequantized_path, output_name)

            else:
                print("Invalid choice! Please enter 1-4.")


# # --------------------------
# # Main
# # --------------------------
if __name__ == "__main__":
    CLI().run()
