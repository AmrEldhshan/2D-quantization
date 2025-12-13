import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import sys
from io import StringIO



# --------------------------
# PredictiveCoding2D class
# --------------------------
class PredictiveCoding2D:
    # These are the default 2-bit parameters from your table
    Q_BINS_2BIT = np.array([-3.0, -0.5, 2.5, 4.5])
    Q_DEQUANT_VALUES_2BIT = np.array([-4.0, -1.0, 2.0, 5.0])
    ADAPTIVE_LEVELS_2BIT = 4
    ADAPTIVE_BITS_2BIT = 2

    def __init__(self, num_bits=None):
        # Set number of bits (default to 2 if not specified)
        if num_bits is None:
            self.num_bits = self.ADAPTIVE_BITS_2BIT
        else:
            self.num_bits = num_bits

        # Initialize quantization parameters that will be created based on num_bits
        self.Q_BINS = None
        self.Q_DEQUANT_VALUES = None
        self.ADAPTIVE_LEVELS = None
        self.quantization_params = None

        # Generate quantization tables based on number of bits
        self._generate_quantization_tables()

    def _generate_quantization_tables(self):
        """Generate quantization tables based on the number of bits"""
        if self.num_bits == 2:
            # Use the predefined 2-bit table
            self.Q_BINS = self.Q_BINS_2BIT
            self.Q_DEQUANT_VALUES = self.Q_DEQUANT_VALUES_2BIT
            self.ADAPTIVE_LEVELS = self.ADAPTIVE_LEVELS_2BIT
        else:
            # For other bit depths, create uniform quantization tables
            # This is a simple implementation - you might want to customize it
            levels = 2 ** self.num_bits

            # Create uniform quantization bins from -255 to 255 (covering full error range)
            # You can adjust this range based on your needs
            min_error = -255.0
            max_error = 255.0

            # Create uniform quantization bins
            step = (max_error - min_error) / levels
            self.Q_BINS = np.array([min_error + i * step for i in range(1, levels)])

            # Create reconstruction values (midpoints of bins)
            self.Q_DEQUANT_VALUES = np.array([min_error + (i + 0.5) * step for i in range(levels)])

            self.ADAPTIVE_LEVELS = levels

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

                    # JPEG-LS predictor
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
    # Adaptive Quantization - per channel
    # --------------------------
    def quantize_dequantize(self, error):
        """Quantize and dequantize each channel"""
        levels = self.ADAPTIVE_LEVELS
        h, w, c = error.shape

        quantized = np.zeros_like(error, dtype=np.int32)
        dequantized = np.zeros_like(error, dtype=np.float32)

        # Store parameters for each channel
        self.quantization_params = []

        # Process each channel independently
        for ch in range(c):
            error_ch = error[:, :, ch]

            # 1. Quantization (Q) using bins
            quantized_ch = np.digitize(error_ch, self.Q_BINS)
            quantized_ch = np.clip(quantized_ch, 0, levels - 1)
            quantized[:, :, ch] = quantized_ch

            # 2. De-quantization (Q^-1)
            dequantized[:, :, ch] = self.Q_DEQUANT_VALUES[quantized_ch]

            # Store parameters for this channel
            self.quantization_params.append({
                'channel': ch,
                'e_min': error_ch.min(),
                'e_max': error_ch.max(),
                'bins': self.Q_BINS.copy(),
                'dequant_values': self.Q_DEQUANT_VALUES.copy(),
                'levels': levels,
                'num_bits': self.num_bits
            })

        return quantized, dequantized

    # --------------------------
    # Save image helper
    # --------------------------
    def save_vis(self, img, path):
        img_clipped = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img_clipped).save(path)

    # --------------------------
    # Save quantized as image visualized
    # --------------------------
    def save_quantized_image(self, quantized, output_path):
        """Save quantized data as a visualizable image (per channel)"""
        quantized_img = np.zeros_like(quantized, dtype=np.uint8)

        for ch in range(quantized.shape[2]):
            ch_data = quantized[:, :, ch]
            if ch_data.max() > ch_data.min():
                quantized_img[:, :, ch] = ((ch_data - ch_data.min()) /
                                           (ch_data.max() - ch_data.min()) * 255).astype(np.uint8)

        Image.fromarray(quantized_img).save(output_path)
        print(f"Quantized image saved to: {output_path}")
        return quantized_img

    # --------------------------
    # Save quantized as npz binary data - per channel
    # --------------------------
    def save_quantized_npz(self, quantized, quant_params, output_path):
        """Save quantized data, prediction, and parameters per channel as .npz file"""
        # Store channel-specific parameters
        save_dict = {
            'quantized': quantized,
            'predicted_data': self.predicted,
            'num_channels': quantized.shape[2],
            'num_bits': self.num_bits
        }

        # Store parameters for each channel
        for ch, params in enumerate(quant_params):
            save_dict[f'e_min_ch{ch}'] = params['e_min']
            save_dict[f'e_max_ch{ch}'] = params['e_max']
            save_dict[f'bins_ch{ch}'] = params['bins']
            save_dict[f'dequant_values_ch{ch}'] = params['dequant_values']
            save_dict[f'levels_ch{ch}'] = params['levels']
            save_dict[f'num_bits_ch{ch}'] = params['num_bits']

        np.savez_compressed(output_path, **save_dict)
        print(f"Quantized data (npz) saved to: {output_path}")

    # --------------------------
    # Load quantized from npz - per channel
    # --------------------------
    def load_quantized_npz(self, quantized_path):
        """Load quantized data, prediction, and parameters from .npz file"""
        data = np.load(quantized_path)
        quantized = data['quantized']
        predicted = data['predicted_data']
        num_channels = data['num_channels'].item()
        num_bits = data['num_bits'].item()

        # Set up the quantization tables based on loaded bits
        self.num_bits = num_bits
        self._generate_quantization_tables()

        quant_params = []
        for ch in range(num_channels):
            quant_params.append({
                'channel': ch,
                'e_min': data[f'e_min_ch{ch}'].item(),
                'e_max': data[f'e_max_ch{ch}'].item(),
                'bins': data[f'bins_ch{ch}'],
                'dequant_values': data[f'dequant_values_ch{ch}'],
                'levels': data[f'levels_ch{ch}'].item(),
                'num_bits': data[f'num_bits_ch{ch}'].item()
            })

        return quantized, quant_params, predicted

    # --------------------------
    # Dequantize - PER CHANNEL
    # --------------------------
    def dequantize(self, quantized, quant_params):
        """Dequantize using the dequantization values from parameters"""
        h, w, c = quantized.shape
        dequantized = np.zeros_like(quantized, dtype=np.float32)

        for ch in range(c):
            # Use the dequantization values from the saved parameters
            dequant_values = quant_params[ch]['dequant_values']
            dequantized[:, :, ch] = dequant_values[quantized[:, :, ch]]

        return dequantized

    # --------------------------
    # Compression Ratio
    # --------------------------
    def calculate_compression_ratio(self, original_image):
        if not isinstance(original_image, np.ndarray):
            original_image = self.original if hasattr(self, 'original') else None
            if original_image is None:
                return None, 0, 0

        original_bpp = 8.0 * 3  # 8 bits per channel * 3 channels
        compressed_bpp = float(self.num_bits * 3)  # user bits per channel * 3 channels

        compression_ratio = original_bpp / compressed_bpp

        # Estimated Size
        original_size_bytes = original_image.size * (8.0 / 8.0)  # 8 bits per pixel
        compressed_size_bytes = original_image.size * (self.num_bits / 8.0)  # user bits per pixel

        ratio_string = f"{round(compression_ratio, 2)}:1"

        return ratio_string, original_size_bytes, compressed_size_bytes

    # --------------------------
    # Compress
    # --------------------------
    def compress(self, img_path, num_bits=None):
        # Update bits if specified
        if num_bits is not None:
            self.num_bits = num_bits
            self._generate_quantization_tables()

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

        # Decompress from the generated NPZ file
        self.decompress_single_npz(quantized_npz_path, "6_decompressed.png")

        # --- Calculate and Display Compression Ratio ---
        ratio_str, original_size, compressed_size = self.calculate_compression_ratio(original)
        if ratio_str:
            print("\n" + "=" * 60)
            print(f"Compression Ratio (24 bits / {self.num_bits*3} bits): {ratio_str}")
            print(f"Bits per channel: {self.num_bits} bits")
            print(f"Original size (estimated): {round(original_size / 1024, 2)} KB")
            print(f"Compressed data size (estimated): {round(compressed_size / 1024, 2)} KB")
            print("=" * 60)

        print("\n" + "=" * 60)
        print(f"COMPRESSION COMPLETE! Files saved in 'output' folder (using {self.num_bits} bits/channel):")
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
    # Decompress from single npz file
    # --------------------------
    def decompress_single_npz(self, quantized_npz_path, output_name="decompressed_from_npz.png"):
        """Decompress using only the quantized npz file containing predicted data and indices."""

        # Load quantized data, parameters, AND predicted data
        quantized, quant_params, predicted = self.load_quantized_npz(quantized_npz_path)

        # Dequantize the error
        dequantized = self.dequantize(quantized, quant_params)

        # Reconstruct: Predicted + Dequantized Error
        decompressed = predicted + dequantized

        # Save
        folder = "output"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, output_name)
        self.save_vis(decompressed, output_path)

        print(f"\nDecompressed from single npz file saved as: {output_path}")
        print(f"Using {self.num_bits} bits per channel")
        return decompressed

    # --------------------------
    # Decompress from images (Original method)
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
    # Print Quantization Table - PER CHANNEL
    # --------------------------
    def print_quantization_table(self):
        N = self.num_bits
        levels = self.ADAPTIVE_LEVELS

        # Ranges need to be explicitly calculated from BINS
        bin_lowers = np.insert(self.Q_BINS, 0, -np.inf)
        bin_uppers = np.append(self.Q_BINS, np.inf)

        header_width = 110

        # Print table for each channel
        for ch_params in self.quantization_params:
            ch = ch_params['channel']
            channel_names = ['RED', 'GREEN', 'BLUE']

            print("\n" + "=" * header_width)
            print(f"QUANTIZATION TABLE - {channel_names[ch]} CHANNEL ({N} BITS, {levels} LEVELS)")
            print("=" * header_width)
            print(f"Error Min (Observed): {ch_params['e_min']:.4f} | Error Max (Observed): {ch_params['e_max']:.4f} | Levels: {levels}")
            print("-" * header_width)

            # Header for the table
            print(f"| {'Index (Q)':<7} | {'Binary Code':<15} | {'De-quantized Value (Q^-1)':<25} | {'Input Error Range Lower, Upper':<45} |")
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


# # --------------------------
# # CLI class
# # --------------------------
class CLI:
    def run(self):
        while True:
            print("\n" + "=" * 60)
            print("ADAPTIVE PREDICTIVE CODING SYSTEM")
            print("Per-Channel Processing Enabled")
            print("=" * 60)
            print("1. Compress image (Specify bits per channel)")
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

                # Ask for number of bits with default value
                bits_input = input("Enter number of bits per channel (default=2): ").strip()
                if bits_input == "":
                    num_bits = 2  # Default value
                else:
                    try:
                        num_bits = int(bits_input)
                        if num_bits <= 0 or num_bits > 8:
                            print("Warning: Bits should be between 1 and 8. Using default (2).")
                            num_bits = 2
                    except ValueError:
                        print("Invalid input. Using default (2 bits).")
                        num_bits = 2

                print(f"\nUsing {num_bits} bits per channel...")

                pc = PredictiveCoding2D(num_bits=num_bits)
                pc.compress(img_path)
                pc.print_quantization_table()

            elif choice == "2":  # Single NPZ Decompression
                print("\n--- Decompress from single NPZ file (Option 2) ---")
                quantized_npz_path = input("Enter quantized NPZ file path (e.g., output/4_quantized.npz): ").strip()

                if not os.path.exists(quantized_npz_path):
                    print("Error: NPZ file not found!")
                    continue

                output_name = input("Enter output filename (default=decompressed_from_npz.png): ").strip()
                output_name = output_name if output_name else "decompressed_from_npz.png"

                # Create instance without specifying bits (will be loaded from file)
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

                # Create instance with default bits (won't matter for this method)
                pc = PredictiveCoding2D()
                pc.decompress(predicted_path, dequantized_path, output_name)

            else:
                print("Invalid choice! Please enter 1-4.")


# # --------------------------
# # Main Application Entry Point
# # --------------------------
# if __name__ == "__main__":
#     CLI().run()



# --------------------------
# ImageCompressionGUI class
# --------------------------
class ImageCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Predictive Coding System")
        self.root.geometry("1200x800")

        # Initialize compressor
        self.pc = None
        self.last_compression_result = None

        # Image display variables
        self.image_labels = []

        # Setup GUI
        self.setup_gui()

        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)

    def setup_gui(self):
        """Setup the main GUI layout"""
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_container,
                               text="ADAPTIVE PREDICTIVE CODING SYSTEM",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Create tabs
        self.compress_tab = ttk.Frame(self.notebook)
        self.decompress_tab = ttk.Frame(self.notebook)
        self.images_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.compress_tab, text="Compression")
        self.notebook.add(self.decompress_tab, text="Decompression")
        self.notebook.add(self.images_tab, text="View Images")
        self.notebook.add(self.results_tab, text="Results")

        # Setup each tab
        self.setup_compress_tab()
        self.setup_decompress_tab()
        self.setup_images_tab()
        self.setup_results_tab()

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_container,
                                          variable=self.progress_var,
                                          maximum=100,
                                          mode='determinate')
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 5))

        # Status label
        self.status_label = ttk.Label(main_container, text="Ready")
        self.status_label.grid(row=3, column=0, columnspan=3, sticky=tk.W)

        # Console output
        console_frame = ttk.LabelFrame(main_container, text="Console Output", padding="5")
        console_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console_text = scrolledtext.ScrolledText(console_frame,
                                                     height=8,
                                                     font=('Courier', 9))
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        main_container.rowconfigure(4, weight=1)

    def setup_compress_tab(self):
        """Setup compression tab"""
        input_frame = ttk.LabelFrame(self.compress_tab, text="Compression Input", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        input_frame.columnconfigure(1, weight=1)

        # Image selection
        ttk.Label(input_frame, text="Image Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.image_path_var = tk.StringVar()
        self.image_entry = ttk.Entry(input_frame, textvariable=self.image_path_var, width=50)
        self.image_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))

        browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_image)
        browse_btn.grid(row=0, column=2, pady=5)

        # Number of bits input
        ttk.Label(input_frame, text="Number of Bits per Channel:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bits_var = tk.StringVar(value="2")
        self.bits_entry = ttk.Entry(input_frame, textvariable=self.bits_var, width=10)
        self.bits_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        ttk.Label(input_frame, text="(Default: 2)").grid(row=1, column=2, sticky=tk.W, pady=5)

        # Output folder
        ttk.Label(input_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_folder_var = tk.StringVar(value="output")
        self.output_entry = ttk.Entry(input_frame, textvariable=self.output_folder_var, width=50)
        self.output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        output_browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_output_folder)
        output_browse_btn.grid(row=2, column=2, pady=5)

        # Info about 2-bit default
        info_frame = ttk.Frame(input_frame)
        info_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))

        ttk.Label(info_frame, text="Note:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(info_frame, text="2 bits uses fixed adaptive quantizer table",
                 font=('Arial', 9)).pack(side=tk.LEFT)

        # Compress button
        self.compress_btn = ttk.Button(input_frame,
                                      text="Start Compression",
                                      command=self.start_compression)
        self.compress_btn.grid(row=4, column=0, columnspan=3, pady=(20, 0))

        self.compress_tab.columnconfigure(0, weight=1)
        self.compress_tab.rowconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)

    def setup_decompress_tab(self):
        """Setup decompression tab - with two sections"""
        # Create a frame with two columns
        main_frame = ttk.Frame(self.decompress_tab)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        self.decompress_tab.columnconfigure(0, weight=1)
        self.decompress_tab.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left side: Decompress from NPZ (Recommended)
        npz_frame = ttk.LabelFrame(main_frame, text="Decompress from NPZ File (Recommended)", padding="10")
        npz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5), pady=5)
        npz_frame.columnconfigure(1, weight=1)

        ttk.Label(npz_frame, text="NPZ File Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.npz_path_var = tk.StringVar()
        self.npz_entry = ttk.Entry(npz_frame, textvariable=self.npz_path_var, width=40)
        self.npz_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))

        browse_npz_btn = ttk.Button(npz_frame, text="Browse", command=self.browse_npz_file)
        browse_npz_btn.grid(row=0, column=2, pady=5)

        ttk.Label(npz_frame, text="Output Filename:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.npz_output_var = tk.StringVar(value="decompressed_from_npz.png")
        self.npz_output_entry = ttk.Entry(npz_frame, textvariable=self.npz_output_var, width=40)
        self.npz_output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.decompress_npz_btn = ttk.Button(npz_frame,
                                           text="Decompress from NPZ",
                                           command=self.start_decompress_npz)
        self.decompress_npz_btn.grid(row=2, column=0, columnspan=3, pady=(15, 0))

        # Right side: Decompress from Images (Legacy)
        img_frame = ttk.LabelFrame(main_frame, text="Decompress from Images (Legacy)", padding="10")
        img_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0), pady=5)
        img_frame.columnconfigure(1, weight=1)

        ttk.Label(img_frame, text="Predicted Image:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.pred_path_var = tk.StringVar()
        self.pred_entry = ttk.Entry(img_frame, textvariable=self.pred_path_var, width=40)
        self.pred_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))

        browse_pred_btn = ttk.Button(img_frame, text="Browse", command=lambda: self.browse_file(self.pred_path_var))
        browse_pred_btn.grid(row=0, column=2, pady=5)

        ttk.Label(img_frame, text="Dequantized Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.deq_path_var = tk.StringVar()
        self.deq_entry = ttk.Entry(img_frame, textvariable=self.deq_path_var, width=40)
        self.deq_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))

        browse_deq_btn = ttk.Button(img_frame, text="Browse", command=lambda: self.browse_file(self.deq_path_var))
        browse_deq_btn.grid(row=1, column=2, pady=5)

        ttk.Label(img_frame, text="Output Filename:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.img_output_var = tk.StringVar(value="decompressed.png")
        self.img_output_entry = ttk.Entry(img_frame, textvariable=self.img_output_var, width=40)
        self.img_output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.decompress_img_btn = ttk.Button(img_frame,
                                           text="Decompress from Images",
                                           command=self.start_decompress_images)
        self.decompress_img_btn.grid(row=3, column=0, columnspan=3, pady=(15, 0))

    def setup_images_tab(self):
        """Setup images display tab"""
        # Create a scrollable frame
        main_frame = ttk.Frame(self.images_tab)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        self.images_tab.columnconfigure(0, weight=1)
        self.images_tab.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Create image labels in a 2x3 grid
        image_titles = [
            "1. Original Image",
            "2. Predicted Image",
            "3. Error Image",
            "4. Quantized Image",
            "5. De-quantized Image",
            "6. Decompressed Image"
        ]

        self.image_labels = []
        self.image_frames = []  # Store frames for resizing
        for i, title in enumerate(image_titles):
            # Create a frame for each image
            frame = ttk.LabelFrame(scrollable_frame, text=title, padding="10")

            # Calculate row and column (2 columns, 3 rows)
            row = i // 2
            col = i % 2

            frame.grid(row=row, column=col, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Create a label for the image (use tk.Label for better image support)
            label = tk.Label(frame, text="No image available", bg='white', relief=tk.SUNKEN)
            label.pack(expand=True, fill='both', padx=5, pady=5)
            self.image_labels.append(label)

            # Configure grid weights
            scrollable_frame.columnconfigure(col, weight=1, uniform="col")
            scrollable_frame.rowconfigure(row, weight=1, uniform="row")

            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)

    def load_and_display_images(self):
        """Load and display all generated images"""
        try:
            # Load images from output folder
            image_files = [
                "1_original.png",
                "2_predicted.png",
                "3_error.png",
                "4_quantized.png",
                "5_dequantized.png",
                "6_decompressed.png"
            ]

            for i, filename in enumerate(image_files):
                filepath = os.path.join("output", filename)
                if os.path.exists(filepath):
                    try:
                        # Open and resize the image
                        img = Image.open(filepath)

                        # Calculate target size (max 300x300 while maintaining aspect ratio)
                        max_width, max_height = 300, 300
                        img_width, img_height = img.size

                        # Calculate scaling factor
                        width_ratio = max_width / img_width
                        height_ratio = max_height / img_height
                        scale = min(width_ratio, height_ratio)

                        # Calculate new size
                        new_width = int(img_width * scale)
                        new_height = int(img_height * scale)

                        # Resize image
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        # Convert to PhotoImage
                        photo = ImageTk.PhotoImage(img)

                        # Update label with image
                        self.image_labels[i].config(image=photo, text="")
                        self.image_labels[i].image = photo  # Keep reference to prevent garbage collection

                    except Exception as e:
                        self.image_labels[i].config(image="", text=f"Error loading {filename}\n{str(e)}", bg='white')
                        self.log_to_console(f"Error loading {filename}: {str(e)}")
                else:
                    self.image_labels[i].config(image="", text=f"{filename} not found", bg='white')

        except Exception as e:
            self.log_to_console(f"Error loading images: {str(e)}")

    def setup_results_tab(self):
        """Setup results tab"""
        text_frame = ttk.Frame(self.results_tab)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        self.results_tab.columnconfigure(0, weight=1)
        self.results_tab.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(text_frame,
                                                     wrap=tk.WORD,
                                                     font=('Courier', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def browse_image(self):
        """Browse for image file"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.image_path_var.set(filename)

    def browse_npz_file(self):
        """Browse for NPZ file"""
        filename = filedialog.askopenfilename(
            title="Select NPZ File",
            filetypes=[
                ("NPZ files", "*.npz"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.npz_path_var.set(filename)

    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)

    def browse_file(self, var):
        """Browse for any file"""
        filename = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)

    def start_compression(self):
        """Start compression"""
        # Get inputs
        image_path = self.image_path_var.get()
        bits_str = self.bits_var.get()
        output_folder = self.output_folder_var.get()

        # Validate inputs
        if not image_path:
            messagebox.showerror("Error", "Please select an image file!")
            return

        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file not found: {image_path}")
            return

        try:
            num_bits = int(bits_str) if bits_str else 2
            if num_bits <= 0:
                messagebox.showerror("Error", "Number of bits must be positive!")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of bits must be an integer!")
            return

        # Create output folder if it doesn't exist
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Disable button and update status
        self.compress_btn.config(state='disabled')
        self.update_progress(f"Starting compression with {num_bits} bits/channel...", 10)

        try:
            # Create PredictiveCoding2D instance with specified bits
            self.pc = PredictiveCoding2D(num_bits=num_bits)

            # Capture console output
            original_stdout = sys.stdout
            sys.stdout = StringIO()

            # Run compression
            self.update_progress("Running compression algorithm...", 30)
            self.pc.compress(image_path)

            # Get captured output
            captured_output = sys.stdout.getvalue()
            sys.stdout = original_stdout

            # Display captured output
            if captured_output:
                self.log_to_console(captured_output)

            # Update progress
            self.update_progress("Compression completed!", 100)

            # Load and display images
            self.load_and_display_images()

            # Update results tab
            self.update_results_tab()

            # Show success message
            messagebox.showinfo("Success", "Compression completed successfully!")

            # Switch to images tab
            self.notebook.select(self.images_tab)

        except Exception as e:
            # Restore stdout if there's an error
            sys.stdout = original_stdout if 'original_stdout' in locals() else sys.stdout
            messagebox.showerror("Error", f"Compression failed: {str(e)}")
            self.log_to_console(f"✗ Compression failed: {str(e)}")
        finally:
            self.compress_btn.config(state='normal')
            self.progress_var.set(0)
            self.status_label.config(text="Ready")

    def start_decompress_npz(self):
        """Start decompression from NPZ"""
        if not self.npz_path_var.get():
            messagebox.showerror("Error", "Please select an NPZ file")
            return

        self.decompress_npz_btn.config(state='disabled')
        self.update_progress("Decompressing from NPZ...", 10)

        try:
            output_name = self.npz_output_var.get()
            if not output_name.endswith('.png'):
                output_name += '.png'

            # Create PredictiveCoding2D instance
            self.pc = PredictiveCoding2D()

            # Capture console output
            original_stdout = sys.stdout
            sys.stdout = StringIO()

            self.update_progress("Running decompression algorithm...", 30)
            decompressed = self.pc.decompress_single_npz(self.npz_path_var.get(), output_name)

            # Get captured output
            captured_output = sys.stdout.getvalue()
            sys.stdout = original_stdout

            # Display captured output
            if captured_output:
                self.log_to_console(captured_output)

            if decompressed is None:
                messagebox.showerror("Decompression Error", "Decompression failed!")
                self.update_progress("Decompression failed", 0)
                return

            # Update result
            self.update_progress("Decompression completed!", 100)

            # Show success message
            messagebox.showinfo("Success", "Decompression from NPZ completed successfully!")

            # Refresh images display
            self.load_and_display_images()

            # Update results tab
            self.update_results_tab()

        except Exception as e:
            # Restore stdout if there's an error
            sys.stdout = original_stdout if 'original_stdout' in locals() else sys.stdout
            messagebox.showerror("Error", f"Decompression failed: {str(e)}")
            self.log_to_console(f"✗ Decompression failed: {str(e)}")
        finally:
            self.decompress_npz_btn.config(state='normal')
            self.progress_var.set(0)
            self.status_label.config(text="Ready")

    def start_decompress_images(self):
        """Start decompression from images"""
        if not self.pred_path_var.get() or not self.deq_path_var.get():
            messagebox.showerror("Error", "Please select both predicted and dequantized images")
            return

        self.decompress_img_btn.config(state='disabled')
        self.update_progress("Decompressing from images...", 10)

        try:
            output_name = self.img_output_var.get()
            if not output_name.endswith('.png'):
                output_name += '.png'

            # Create PredictiveCoding2D instance
            self.pc = PredictiveCoding2D()

            # Capture console output
            original_stdout = sys.stdout
            sys.stdout = StringIO()

            self.update_progress("Running decompression algorithm...", 30)
            decompressed = self.pc.decompress(
                self.pred_path_var.get(),
                self.deq_path_var.get(),
                output_name
            )

            # Get captured output
            captured_output = sys.stdout.getvalue()
            sys.stdout = original_stdout

            # Display captured output
            if captured_output:
                self.log_to_console(captured_output)

            if decompressed is None:
                messagebox.showerror("Decompression Error", "Decompression failed!")
                self.update_progress("Decompression failed", 0)
                return

            # Update result
            self.update_progress("Decompression completed!", 100)

            # Show success message
            messagebox.showinfo("Success", "Decompression from images completed successfully!")

            # Refresh images display
            self.load_and_display_images()

            # Update results tab
            self.update_results_tab()

        except Exception as e:
            # Restore stdout if there's an error
            sys.stdout = original_stdout if 'original_stdout' in locals() else sys.stdout
            messagebox.showerror("Error", f"Decompression failed: {str(e)}")
            self.log_to_console(f"✗ Decompression failed: {str(e)}")
        finally:
            self.decompress_img_btn.config(state='normal')
            self.progress_var.set(0)
            self.status_label.config(text="Ready")

    def load_and_display_images(self):
        """Load and display all generated images"""
        try:
            # Load images from output folder
            image_files = [
                "1_original.png",
                "2_predicted.png",
                "3_error.png",
                "4_quantized.png",
                "5_dequantized.png",
                "6_decompressed.png"
            ]

            for i, filename in enumerate(image_files):
                filepath = os.path.join("output", filename)
                if os.path.exists(filepath):
                    try:
                        img = Image.open(filepath)
                        # Calculate aspect ratio preserving size
                        width, height = img.size
                        max_size = 300
                        ratio = min(max_size/width, max_size/height)
                        new_size = (int(width * ratio), int(height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)

                        photo = ImageTk.PhotoImage(img)
                        self.image_labels[i].config(image=photo, text="")
                        self.image_labels[i].image = photo  # Keep reference
                    except Exception as e:
                        self.image_labels[i].config(image="", text=f"Error loading {filename}")
                        self.log_to_console(f"Error loading {filename}: {str(e)}")
                else:
                    self.image_labels[i].config(image="", text=f"{filename} not found")

        except Exception as e:
            self.log_to_console(f"Error loading images: {str(e)}")

    def update_progress(self, message, value=None):
        """Update progress bar and status"""
        self.status_label.config(text=message)
        if value is not None:
            self.progress_var.set(value)
        self.log_to_console(message)
        self.root.update_idletasks()

    def log_to_console(self, message):
        """Add message to console"""
        self.console_text.insert(tk.END, message)
        self.console_text.see(tk.END)
        self.console_text.update_idletasks()


    def update_results_tab(self):
        """Update the results tab with compression information"""
        if self.pc is not None and hasattr(self.pc, 'quantization_params') and self.pc.quantization_params:
            # Get quantization table text from the pc object
            quant_text = self.get_quantization_table_text()

            # Add compression ratio info
            if hasattr(self.pc, 'original'):
                ratio_str, original_size, compressed_size = self.pc.calculate_compression_ratio(self.pc.original)

                results_text = "=" * 80 + "\n"
                results_text += "COMPRESSION RESULTS\n"
                results_text += "=" * 80 + "\n\n"

                results_text += f"BITS PER CHANNEL: {self.pc.num_bits}\n"
                results_text += f"COMPRESSION RATIO: {ratio_str}\n"
                results_text += f"ORIGINAL SIZE: {round(original_size / 1024, 2):.2f} KB\n"
                results_text += f"COMPRESSED SIZE: {round(compressed_size / 1024, 2):.2f} KB\n\n"

                results_text += quant_text

                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, results_text)
        else:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END,
                "No compression results available. Run compression first.\n\n"
                "After compression, this tab will show:\n"
                "• Quantization tables for each channel\n"
                "• Compression ratio information\n"
                "• File sizes\n"
                "• Output file list")

    def get_quantization_table_text(self):
        """Generate quantization table text from pc object"""
        if not hasattr(self.pc, 'quantization_params') or not self.pc.quantization_params:
            return "No quantization parameters available."

        text = "\n" + "=" * 80 + "\n"
        text += "QUANTIZATION TABLES\n"
        text += "=" * 80 + "\n\n"

        # For each channel
        for ch_params in self.pc.quantization_params:
            ch = ch_params['channel']
            channel_names = ['RED', 'GREEN', 'BLUE']

            text += f"{channel_names[ch]} CHANNEL:\n"
            text += "-" * 40 + "\n"
            text += f"Observed Error Min: {ch_params['e_min']:.4f}\n"
            text += f"Observed Error Max: {ch_params['e_max']:.4f}\n"
            text += f"Levels: {ch_params['levels']}\n"
            text += f"Bits: {ch_params['num_bits']}\n\n"

            # Get bins and dequant values
            bins = ch_params.get('bins', self.pc.Q_BINS)
            dequant_values = ch_params.get('dequant_values', self.pc.Q_DEQUANT_VALUES)
            levels = ch_params['levels']

            # Create bin boundaries
            bin_lowers = np.insert(bins, 0, -np.inf)
            bin_uppers = np.append(bins, np.inf)

            # Create table header
            text += f"{'Index':<8} {'Binary':<10} {'Dequant Value':<15} {'Range':<30}\n"
            text += "-" * 63 + "\n"

            # Add each level
            for i in range(levels):
                index = i
                binary_code = format(i, f'0{self.pc.num_bits}b')

                lower = bin_lowers[i]
                upper = bin_uppers[i]

                # Format the range string
                lower_str = f"{lower:.4f}" if lower != -np.inf else "-inf"
                upper_str = f"{upper:.4f}" if upper != np.inf else "inf"
                range_str = f"{lower_str}, {upper_str}"

                dequant_value = dequant_values[i] if i < len(dequant_values) else 0

                text += f"{index:<8} {binary_code:<10} {dequant_value:<15.4f} {range_str:<30}\n"

            text += "\n"

        return text


# --------------------------
# CLI Menu (to choose between GUI and CLI)
# --------------------------
def run_cli_menu():
    """CLI menu to choose between GUI and CLI modes"""
    while True:
        print("\n" + "=" * 60)
        print("ADAPTIVE PREDICTIVE CODING SYSTEM")
        print("Per-Channel Processing")
        print("=" * 60)
        print("1. Run GUI Interface")
        print("2. Run Command Line Interface")
        print("3. Exit")
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "3":
            print("Exiting...")
            return
        elif choice == "1":
            run_gui()
            break
        elif choice == "2":
            run_command_line()
            break
        else:
            print("Invalid choice! Please enter 1-3.")

def run_gui():
    """Run the GUI application"""
    root = tk.Tk()
    app = ImageCompressionGUI(root)
    root.mainloop()

def run_command_line():
    """Run the command line interface"""
    cli = CLI()
    cli.run()


# --------------------------
# Modified Main Application Entry Point
# --------------------------
if __name__ == "__main__":
    run_cli_menu()
