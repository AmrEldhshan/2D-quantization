import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import struct  # Required for binary packing
class PredictiveCoding2D:
    def __init__(self, num_bits=2, use_color=True):
        """
        Initialize the system with user-defined bits.
        Default is 2 bits (4 levels: 00, 01, 10, 11).
        """
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.use_color = use_color

        # Store quantizer settings for the binary header
        self.quantizers_info = []

        # Image Buffers
        self.original = None
        self.predicted = None
        self.error = None
        self.quantized = None     # Stores indices (0, 1, 2, 3)
        self.dequantized = None   # Stores actual values (e.g., -10.5, 5.2)
        self.decompressed = None  # Reconstructed image

    def adaptive_predictor(self, A, B, C):
        """
        [cite_start]Adaptive 2D Predictor Logic from Lecture Slide 13 [cite: 109, 110]
        """
        min_AC = min(A, C)
        max_AC = max(A, C)

        if B <= min_AC:
            return max_AC
        elif B >= max_AC:
            return min_AC
        else:
            return A + C - B

    def create_uniform_quantizer(self, min_val, max_val):
        """
        [cite_start]Creates a dynamic uniform quantizer based on error range [cite: 47]
        """
        quantizer = []
        # Avoid division by zero
        range_val = max_val - min_val
        if abs(range_val) < 1e-10:
            range_val = 1.0

        step = range_val / self.num_levels

        for i in range(self.num_levels):
            lower = min_val + i * step
            upper = min_val + (i + 1) * step
            # Reconstruction value is the midpoint
            dequant = (lower + upper) / 2.0

            quantizer.append({
                'level': i,
                'lower': lower,
                'upper': upper,
                'dequant': dequant
            })
        return quantizer

    def quantize_value(self, value, quantizer):
        """Maps an error value to a specific level index and de-quantized value"""
        # 1. Check boundaries (Optimization)
        if value < quantizer[0]['lower']:
            return quantizer[0]['level'], quantizer[0]['dequant']
        if value >= quantizer[-1]['upper']:
            return quantizer[-1]['level'], quantizer[-1]['dequant']

        # 2. Linear search for the bin
        for q in quantizer:
            if q['lower'] <= value < q['upper']:
                return q['level'], q['dequant']

        # Fallback
        return quantizer[-1]['level'], quantizer[-1]['dequant']

    def compress(self, image_path):
        print("\n" + "="*60)
        print(f"Starting Compression ({self.num_bits} bits)...")

        # --- 1. Load Image ---
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Image not found.")
            return False

        # Convert BGR (OpenCV standard) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_color:
            self.original = img.astype(float)
        else:
            self.original = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
            # Add channel dimension for consistency: (H, W, 1)
            self.original = self.original[:, :, np.newaxis]

        height, width, channels = self.original.shape

        # Initialize Output Arrays
        self.predicted = np.zeros_like(self.original)
        self.error = np.zeros_like(self.original)
        self.quantized = np.zeros_like(self.original, dtype=int)
        self.dequantized = np.zeros_like(self.original)
        self.decompressed = np.zeros_like(self.original)
        self.quantizers_info = [] # Reset

        # --- 2. Process Each Channel Separately ---
        for c in range(channels):
            print(f"Processing Channel {c+1}/{channels}...")
            channel_orig = self.original[:, :, c]

            # --- PASS 1: Range Estimation ---
            # We calculate approximate errors using original pixels to find Min/Max
            temp_errors = []
            # Subsampling (step=4) speeds up estimation without losing much accuracy
            for y in range(0, height, 4):
                for x in range(0, width, 4):
                    if x==0 and y==0: p=128.0
                    elif x==0: p=channel_orig[y-1, x]
                    elif y==0: p=channel_orig[y, x-1]
                    else: p=self.adaptive_predictor(channel_orig[y, x-1], channel_orig[y-1, x], channel_orig[y-1, x-1])
                    temp_errors.append(channel_orig[y, x] - p)

            min_e, max_e = min(temp_errors), max(temp_errors)

            # Create Quantizer Table for this channel
            q_table = self.create_uniform_quantizer(min_e, max_e)
            self.quantizers_info.append({'min': min_e, 'max': max_e})

            # --- PASS 2: Feedback Loop (Real Compression) ---
            # [cite_start]We must predict based on the RECONSTRUCTED (Decompressed) image [cite: 57]
            channel_recon = np.zeros((height, width))

            for y in range(height):
                for x in range(width):
                    # A. Predict
                    if x==0 and y==0: pred = 128.0
                    elif x==0: pred = channel_recon[y-1, x]
                    elif y==0: pred = channel_recon[y, x-1]
                    else:
                        # Use Reconstructed neighbors (Feedback)
                        pred = self.adaptive_predictor(channel_recon[y, x-1], channel_recon[y-1, x], channel_recon[y-1, x-1])

                    self.predicted[y, x, c] = pred

                    # B. Calculate Error
                    real_val = channel_orig[y, x]
                    err = real_val - pred
                    self.error[y, x, c] = err

                    # C. Quantize Error
                    level, deq_err = self.quantize_value(err, q_table)
                    self.quantized[y, x, c] = level      # Store index (0-3)
                    self.dequantized[y, x, c] = deq_err  # Store value (e.g. -12.5)

                    # D. Reconstruct (Feedback)
                    recon_val = np.clip(pred + deq_err, 0, 255)
                    channel_recon[y, x] = recon_val
                    self.decompressed[y, x, c] = recon_val

        # Cleanup dimensions for grayscale
        if not self.use_color:
            self.original = self.original.squeeze()
            self.decompressed = self.decompressed.squeeze()

        print("Compression Logic Complete.")
        return True

    def save_binary_file(self, output_path="output/compressed_data.bin"):
        """
        Saves the '4_Quantized' data as a PACKED BINARY FILE.
        This solves the file size issue.
        """
        if self.quantized is None:
            return

        print(f"\n[Binary] Packing bits to '{output_path}'...")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        h, w = self.quantized.shape[:2]
        c = 3 if self.use_color else 1

        with open(output_path, 'wb') as f:
            # --- HEADER ---
            # 1. Signature 'PC2D'
            f.write(b'PC2D')
            # 2. Metadata: Width(2B), Height(2B), Channels(1B), BitsPerPixel(1B)
            f.write(struct.pack('>HHBB', w, h, c, self.num_bits))

            # 3. Quantizer Ranges (Min/Max for each channel)
            # This is needed so a decoder knows what "Level 0" or "Level 3" actually means
            for q_info in self.quantizers_info:
                f.write(struct.pack('>ff', q_info['min'], q_info['max']))

            # --- DATA BODY ---
            # Flatten the array of indices [0, 3, 1, 2, 0, ...]
            flat_indices = self.quantized.flatten().astype(int)

            bit_buffer = 0
            bits_in_buffer = 0
            byte_array = bytearray()

            # Pack bits (e.g., four 2-bit pixels into one 8-bit byte)
            for level in flat_indices:
                bit_buffer = (bit_buffer << self.num_bits) | level
                bits_in_buffer += self.num_bits

                while bits_in_buffer >= 8:
                    # Extract the top 8 bits
                    byte_val = (bit_buffer >> (bits_in_buffer - 8)) & 0xFF
                    byte_array.append(byte_val)
                    bits_in_buffer -= 8

            # Handle remaining bits (padding)
            if bits_in_buffer > 0:
                byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
                byte_array.append(byte_val)

            f.write(byte_array)

        # Calculate compression stats
        bin_size = os.path.getsize(output_path)
        raw_size = h * w * c  # 1 byte per pixel (8-bit)
        print(f"  Binary File Size: {bin_size/1024:.2f} KB")
        print(f"  Original Raw Size: {raw_size/1024:.2f} KB")
        print(f"  Compression Ratio: {raw_size/bin_size:.2f}:1")

    def save_images(self, output_dir="output"):
        """Saves the 6 visual images required by the assignment"""
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        def save_vis(img, name):
            if img is None: return
            # Normalize min-max to 0-255 for visualization
            _min, _max = img.min(), img.max()
            if _max > _min:
                vis = ((img - _min) / (_max - _min) * 255).astype(np.uint8)
            else:
                vis = img.astype(np.uint8)

            # Convert to BGR if color
            if len(vis.shape) == 3:
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(output_dir, name), vis)

        print(f"\n[Visuals] Saving PNGs to '{output_dir}'...")

        # 1. Original
        save_vis(self.original, "1_original.png")
        # 2. Predicted
        save_vis(self.predicted, "2_predicted.png")
        # 3. Error
        save_vis(self.error, "3_error.png")
        # 4. Quantized (Visual Map)
        save_vis(self.quantized, "4_quantized.png")
        # 5. De-quantized
        save_vis(self.dequantized, "5_dequantized.png")
        # 6. Decompressed (Reconstructed)
        save_vis(self.decompressed, "6_decompressed.png")

def main():
    print("--- 2D Predictive Coding Assignment ---")

    # 1. Setup
    img_path = "input.jpg"  # Make sure this file exists
    if not os.path.exists(img_path):
        print(f"Error: Please create '{img_path}' first.")
        return

    # 2. Get User Input
    try:
        bits = int(input("Enter Bits (1-3, Default 2): ").strip() or "2")
    except:
        bits = 2

    # 3. Run Compression
    coder = PredictiveCoding2D(num_bits=bits, use_color=True)
    if coder.compress(img_path):
        # Save Visual Images (Requirements)
        coder.save_images()


if __name__ == "__main__":
    main()
