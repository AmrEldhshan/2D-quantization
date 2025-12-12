import numpy as np
import os
from PIL import Image


# --------------------------
# PredictiveCoding2D class
# --------------------------
class PredictiveCoding2D:
    # --------------------------
    # Adaptive Quantizer Parameters (FIXED by the image)
    # --------------------------
    # Defined bin boundaries for the *error* ranges (upper bound of each level, except the last)
    # RANGES: (-inf, -3.0], (-3.0, -0.5], (-0.5, 2.5], (2.5, 4.5], (4.5, inf)
    Q_BINS = np.array([-3.0, -0.5, 2.5, 4.5])
    # The Q indices (output of quantization)
    Q_INDICES = np.array([0, 1, 2, 3])
    # The Q^-1 values (reconstruction value)
    Q_DEQUANT_VALUES = np.array([-4.0, -1.0, 2.0, 5.0])
    # The number of levels defined by the table is 4
    ADAPTIVE_LEVELS = 4
    ADAPTIVE_BITS = 2  # 4 levels = 2 bits

    def __init__(self, num_bits=None):
        # We enforce 2 bits since the quantization table is fixed at 4 levels
        self.num_bits = self.ADAPTIVE_BITS
        self.quantization_params = None

    # --------------------------
    # Load image
    # --------------------------
    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Image path does not exist.")
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return np.array(img, dtype=np.float32)

    # --------------------------
    # JPEG-LS Predictor (Adaptive 2D Predictor from image)
    # --------------------------
    def predictiveImageGenerate(self, img):
        #
        self.original = img.astype(np.float32)
        h, w, c = self.original.shape
        self.predicted = np.zeros_like(self.original)

        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    # Neighbors
                    A = self.original[y, x - 1, ch] if x > 0 else 0
                    B = self.original[y - 1, x, ch] if y > 0 else 0
                    C = self.original[y - 1, x - 1, ch] if (x > 0 and y > 0) else 0

                    # JPEG-LS predictor (Formula from the image)
                    if B <= min(A, C):
                        P = max(A, C)
                    elif B >= max(A, C):
                        P = min(A, C)
                    else:
                        P = A + C - B

                    self.predicted[y, x, ch] = P

        self.error = self.original - self.predicted
        return self.predicted, self.error

    # --------------------------
    # Adaptive Quantization (Fixed by image table)
    # --------------------------
    def quantize_dequantize(self, error):
        levels = self.ADAPTIVE_LEVELS

        # 1. Quantization (Q) using fixed bins
        # np.digitize returns the index i such that Q_BINS[i-1] < error <= Q_BINS[i]
        quantized = np.digitize(error, self.Q_BINS)

        # Clip to ensure indices are within [0, 3] (4 levels)
        quantized = np.clip(quantized, 0, levels - 1)

        # 2. De-quantization (Q^-1)
        # Use the quantized index to look up the fixed de-quantized value
        dequantized = self.Q_DEQUANT_VALUES[quantized]

        # Store parameters for reference and printout
        self.quantization_params = {
            'e_min': error.min(),
            'e_max': error.max(),
            'step': np.nan,  # FIXED: Changed np.NaN to np.nan
            'levels': levels,
            'num_bits': self.ADAPTIVE_BITS
        }

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
        if quantized.max() > quantized.min():
            quantized_img = ((quantized - quantized.min()) /
                             (quantized.max() - quantized.min()) * 255).astype(np.uint8)
        else:
            quantized_img = np.zeros_like(quantized, dtype=np.uint8)

        Image.fromarray(quantized_img).save(output_path)
        print(f"Quantized image saved to: {output_path}")
        return quantized_img

    # --------------------------
    # Save quantized as npz (binary data)
    # --------------------------
    def save_quantized_npz(self, quantized, quant_params, output_path):
        """Save quantized data, prediction, and fixed parameters as .npz file"""
        np.savez_compressed(
            output_path,
            quantized=quantized,
            predicted_data=self.predicted,  # Predicted data for single-file decompression
            e_min=quant_params['e_min'],
            e_max=quant_params['e_max'],
            levels=quant_params['levels'],
            num_bits=quant_params['num_bits']
        )
        print(f"Quantized data (npz) saved to: {output_path}")

    # --------------------------
    # Load quantized from npz
    # --------------------------
    def load_quantized_npz(self, quantized_path):
        """Load quantized data, prediction, and parameters from .npz file"""
        data = np.load(quantized_path)
        quantized = data['quantized']
        predicted = data['predicted_data']
        quant_params = {
            'e_min': data['e_min'].item(),
            'e_max': data['e_max'].item(),
            'levels': data['levels'].item(),
            'num_bits': data['num_bits'].item()
        }
        return quantized, quant_params, predicted

    # --------------------------
    # Dequantize
    # --------------------------
    def dequantize(self, quantized, quant_params):
        """Dequantize using the fixed Q^-1 values (lookup table implied by the index)"""
        dequantized = self.Q_DEQUANT_VALUES[quantized]
        return dequantized

    # --------------------------
    # Compression Ratio
    # --------------------------
    def calculate_compression_ratio(self, original_image):
        if not isinstance(original_image, np.ndarray):
            original_image = self.original if hasattr(self, 'original') else None
            if original_image is None:
                return None, 0, 0

        original_bpp = 8.0
        compressed_bpp = float(self.num_bits)

        compression_ratio = original_bpp / compressed_bpp

        # Estimated Size
        original_size_bytes = original_image.size * (original_bpp / 8.0)
        compressed_size_bytes = original_image.size * (compressed_bpp / 8.0)

        ratio_string = f"{round(compression_ratio, 2)}:1"

        return ratio_string, original_size_bytes, compressed_size_bytes

    # --------------------------
    # Compress
    # --------------------------
    def compress(self, img_path):
        original = self.load_image(img_path)
        predicted, error = self.predictiveImageGenerate(original)
        quantized, dequantized = self.quantize_dequantize(error)

        folder = "output"
        os.makedirs(folder, exist_ok=True)

        # Save visualization images
        self.save_vis(original, os.path.join(folder, "1_original.png"))
        self.save_vis(predicted, os.path.join(folder, "2_predicted.png"))
        self.save_vis(error + 128, os.path.join(folder, "3_error.png"))

        # Save quantized as IMAGE (visualization)
        quantized_img_path = os.path.join(folder, "4_quantized.png")
        self.save_quantized_image(quantized, quantized_img_path)

        # Save quantized as NPZ (binary data and PREDICTION for decompression)
        quantized_npz_path = os.path.join(folder, "4_quantized.npz")
        self.save_quantized_npz(quantized, self.quantization_params, quantized_npz_path)

        # Save dequantized error image
        dequantized_path = os.path.join(folder, "5_dequantized.png")
        self.save_vis(dequantized, dequantized_path)

        # Decompress from the generated NPZ file (using the new single-file method)
        self.decompress_single_npz(quantized_npz_path, "6_decompressed.png")

        # --- Calculate and Display Compression Ratio ---
        ratio_str, original_size, compressed_size = self.calculate_compression_ratio(original)
        if ratio_str:
            print("\n" + "=" * 60)
            print(f"Compression Ratio (8 bits / {self.num_bits} bits): {ratio_str} (FIXED by Adaptive Quantizer)")
            print(f"Original size (estimated): {round(original_size / 1024, 2)} KB")
            print(f"Compressed data size (estimated): {round(compressed_size / 1024, 2)} KB")
            print("=" * 60)

        print("\n" + "=" * 60)
        print("COMPRESSION COMPLETE! Files saved in 'output' folder:")
        print("=" * 60)
        print("1. Original image:              1_original.png")
        print("2. Predicted image:             2_predicted.png")
        print("3. Error (with +128 offset):    3_error.png")
        print("4. Quantized (IMAGE version):   4_quantized.png")
        print("5. Quantized (NPZ version):     4_quantized.npz <--- Contains ALL data for decompression")
        print("6. Dequantized error:           5_dequantized.png")
        print("7. Final decompressed:          6_decompressed.png")
        print("=" * 60)

    # --------------------------
    # Decompress from single npz file (New and Recommended Method)
    # --------------------------
    def decompress_single_npz(self, quantized_npz_path, output_name="decompressed_from_npz.png"):
        """Decompress using only the quantized npz file containing predicted data and indices."""

        # Load quantized data, parameters, AND predicted data
        quantized, quant_params, predicted = self.load_quantized_npz(quantized_npz_path)

        # Dequantize the error (uses fixed Q^-1 values)
        dequantized = self.dequantize(quantized, quant_params)

        # Reconstruct: Predicted + Dequantized Error
        decompressed = predicted + dequantized

        # Save
        folder = "output"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, output_name)
        self.save_vis(decompressed, output_path)

        print(f"\nDecompressed from single npz file saved as: {output_path}")
        return decompressed

    # --------------------------
    # Decompress from images (Original method - kept for backward compatibility)
    # --------------------------
    def decompress(self, predicted_path, dequantized_path, output_name="decompressed.png"):
        """Decompress using predicted and dequantized images"""
        predicted = self.load_image(predicted_path)
        dequantized = self.load_image(dequantized_path)

        decompressed = predicted + dequantized

        folder = "output"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, output_name)
        self.save_vis(decompressed, output_path)

        print(f"Decompressed image saved as: {output_path}")
        return decompressed

    # --------------------------
    # Print Quantization Table (Adapted for Adaptive/Fixed Table)
    # --------------------------
    def print_quantization_table(self):
        N = self.ADAPTIVE_BITS
        levels = self.ADAPTIVE_LEVELS

        # Ranges need to be explicitly calculated from BINS
        bin_lowers = np.insert(self.Q_BINS, 0, -np.inf)
        bin_uppers = np.append(self.Q_BINS, np.inf)

        header_width = 110

        print("\n" + "=" * header_width)
        print(f"ADAPTIVE QUANTIZATION TABLE (FIXED BY IMAGE) ({N} BITS, {levels} LEVELS)")
        print("=" * header_width)
        print(
            f"Error Min (Observed): {self.quantization_params['e_min']:.4f} | Error Max (Observed): {self.quantization_params['e_max']:.4f} | Fixed Levels: {levels}")
        print("-" * header_width)

        # Header for the table
        print(
            f"| {'Index (Q)':<7} | {'Binary Code':<15} | {'De-quantized Value (Q^-1)':<25} | {'Input Error Range Lower, Upper':<45} |")
        print("-" * header_width)

        for i in range(levels):
            index = i
            binary_code = format(i, f'0{N}b')

            lower = bin_lowers[i]
            upper = bin_uppers[i]

            # Format the range string (handling infinity)
            lower_str = f"{lower:.4f}" if lower != -np.inf else "-inf"
            upper_str = f"{upper:.4f}" if upper != np.inf else "inf"

            # Format to the requested "Lower, Upper" style
            range_str = f"{lower_str}, {upper_str}"

            dequant_value = self.Q_DEQUANT_VALUES[i]

            print(f"| {index:<7} | {binary_code:<15} | {dequant_value:.8f}{'':<17} | {range_str:<45} |")

        print("=" * header_width)


# --------------------------
# CLI class
# --------------------------
class CLI:
    def run(self):
        while True:
            print("\n" + "=" * 50)
            print("ADAPTIVE PREDICTIVE CODING SYSTEM (2 BITS FIXED)")
            print("=" * 50)
            print("1. Compress image (Saves all files + single-NPZ for decompression)")
            print("2. Decompress from single quantized NPZ file (Recommended)")
            print("3. Decompress from predicted image + dequantized image (Legacy)")
            print("4. Exit")
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == "4":
                print("Exiting...")
                return

            if choice == "1":
                img_path = input("Enter image file path (jpg/png): ").strip()
                if not os.path.exists(img_path):
                    print(f"Error: File '{img_path}' not found!")
                    continue

                pc = PredictiveCoding2D()
                pc.compress(img_path)
                pc.print_quantization_table()

            elif choice == "2":  # Single NPZ Decompression
                print("\n--- Decompress from single NPZ file (Option 2) ---")
                quantized_npz_path = input("Enter quantized NPZ file path (e.g., output/4_quantized.npz): ").strip()

                if not os.path.exists(quantized_npz_path):
                    print("Error: NPZ file not found!")
                    continue

                output_name = input("Enter output filename (default=decompressed_from_npz.png): ").strip()
                output_name = output_name if output_name else "decompressed_from_npz_single.png"

                pc = PredictiveCoding2D()
                pc.decompress_single_npz(quantized_npz_path, output_name)

            elif choice == "3":  # Legacy Decompression
                print("\n--- Decompress from images (Option 3) ---")
                predicted_path = input("Enter predicted image path: ").strip()
                dequantized_path = input("Enter dequantized image path: ").strip()

                if not os.path.exists(predicted_path) or not os.path.exists(dequantized_path):
                    print("Error: One or both files not found!")
                    continue

                output_name = input("Enter output filename (default=decompressed.png): ").strip()
                output_name = output_name if output_name else "decompressed.png"

                pc = PredictiveCoding2D()
                pc.decompress(predicted_path, dequantized_path, output_name)

            else:
                print("Invalid choice! Please enter 1-4.")


# --------------------------
# Main Application Entry Point
# --------------------------
if __name__ == "__main__":
    CLI().run()
