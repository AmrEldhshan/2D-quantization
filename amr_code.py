import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from datetime import datetime
import threading
import json
import numpy as np
import struct
import cv2
from PIL import Image, ImageTk


# ============================================================================
# PredictiveCoding2D Class (COMPLETE AND UNMODIFIED from your code)
# ============================================================================
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


# ============================================================================
# GUI Class
# ============================================================================
class ImageCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression System")
        self.root.geometry("1200x800")

        # Initialize compressor
        self.compressor = None
        self.decompressor = None

        # Threading variables
        self.compression_thread = None
        self.decompression_thread = None
        self.is_compressing = False
        self.is_decompressing = False

        # Storage for compression results
        self.last_compression_result = None

        # Image display variables
        self.current_images = None
        self.image_labels = []

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_container,
                               text="IMAGE COMPRESSION SYSTEM",
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

        # Configure weights for console area
        main_container.rowconfigure(4, weight=1)

    def setup_compress_tab(self):
        """Setup compression tab"""
        # Input frame
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

        # Number of bits
        ttk.Label(input_frame, text="Number of Bits (1-8, Default=2):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bits_var = tk.StringVar(value="2")
        self.bits_entry = ttk.Entry(input_frame, textvariable=self.bits_var, width=10)
        self.bits_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Color mode
        ttk.Label(input_frame, text="Color Mode:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.color_mode_var = tk.StringVar(value="Color")
        color_combo = ttk.Combobox(input_frame, textvariable=self.color_mode_var,
                                  values=["Color", "Grayscale"], state="readonly", width=15)
        color_combo.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

        # Output folder
        ttk.Label(input_frame, text="Output Folder:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_folder_var = tk.StringVar(value="output")
        self.output_entry = ttk.Entry(input_frame, textvariable=self.output_folder_var, width=50)
        self.output_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        output_browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_output_folder)
        output_browse_btn.grid(row=3, column=2, pady=5)

        # Compress button
        self.compress_btn = ttk.Button(input_frame,
                                      text="Start Compression",
                                      command=self.start_compression_thread)
        self.compress_btn.grid(row=4, column=0, columnspan=3, pady=(20, 0))

        # Configure weights
        self.compress_tab.columnconfigure(0, weight=1)
        self.compress_tab.rowconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)

    def setup_decompress_tab(self):
        """Setup decompression tab"""
        # Input frame
        input_frame = ttk.LabelFrame(self.decompress_tab, text="Decompression Input", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        input_frame.columnconfigure(1, weight=1)

        # Binary file selection
        ttk.Label(input_frame, text="Compressed Binary File (.bin):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.binary_path_var = tk.StringVar()
        self.binary_entry = ttk.Entry(input_frame, textvariable=self.binary_path_var, width=50)
        self.binary_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))

        browse_bin_btn = ttk.Button(input_frame, text="Browse", command=self.browse_binary_file)
        browse_bin_btn.grid(row=0, column=2, pady=5)

        # Output folder
        ttk.Label(input_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.decompress_output_var = tk.StringVar(value="output_decompressed")
        self.decompress_output_entry = ttk.Entry(input_frame, textvariable=self.decompress_output_var, width=50)
        self.decompress_output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        output_browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_decompress_output_folder)
        output_browse_btn.grid(row=1, column=2, pady=5)

        # Decompress button
        self.decompress_btn = ttk.Button(input_frame,
                                        text="Start Decompression",
                                        command=self.start_decompression_thread)
        self.decompress_btn.grid(row=2, column=0, columnspan=3, pady=(20, 0))

        # Configure weights
        self.decompress_tab.columnconfigure(0, weight=1)
        self.decompress_tab.rowconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)

    def setup_images_tab(self):
        """Setup images display tab"""
        # Create a canvas with scrollbar for images
        canvas_frame = ttk.Frame(self.images_tab)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.images_tab.columnconfigure(0, weight=1)
        self.images_tab.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Create image labels
        image_titles = [
            "1. Original Image",
            "2. Predicted Image",
            "3. Error Image",
            "4. Quantized Image",
            "5. De-quantized Image",
            "6. Decompressed Image"
        ]

        self.image_labels = []
        for i, title in enumerate(image_titles):
            frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding="5")
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

            label = ttk.Label(frame, text="No image available", relief=tk.SUNKEN)
            label.grid(row=0, column=0, padx=5, pady=5)
            self.image_labels.append(label)

            # Configure frame weights
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)

        # Configure scrollable frame grid
        for i in range(2):
            self.scrollable_frame.columnconfigure(i, weight=1)
        for i in range(3):
            self.scrollable_frame.rowconfigure(i, weight=1)

    def setup_results_tab(self):
        """Setup results tab"""
        # Create text widget for results
        text_frame = ttk.Frame(self.results_tab)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Configure grid weights
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

    def browse_binary_file(self):
        """Browse for binary file"""
        filename = filedialog.askopenfilename(
            title="Select Compressed Binary File",
            filetypes=[
                ("Binary files", "*.bin"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.binary_path_var.set(filename)

    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)

    def browse_decompress_output_folder(self):
        """Browse for decompression output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.decompress_output_var.set(folder)

    def start_compression_thread(self):
        """Start compression in a separate thread"""
        if self.is_compressing:
            messagebox.showwarning("Warning", "Compression is already in progress!")
            return

        # Get inputs
        image_path = self.image_path_var.get()
        bits_str = self.bits_var.get()
        color_mode = self.color_mode_var.get()
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
            if num_bits < 1 or num_bits > 8:
                messagebox.showerror("Error", "Number of bits must be between 1 and 8!")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of bits must be an integer!")
            return

        # Create output folder if it doesn't exist
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Disable button and update status
        self.compress_btn.config(state='disabled')
        self.is_compressing = True
        self.update_progress("Starting compression...", 0)

        # Start compression thread
        self.compression_thread = threading.Thread(
            target=self.run_compression,
            args=(image_path, num_bits, color_mode, output_folder)
        )
        self.compression_thread.daemon = True
        self.compression_thread.start()

    def run_compression(self, image_path, num_bits, color_mode, output_folder):
        """Run compression in thread using PredictiveCoding2D class"""
        try:
            # Update progress
            self.update_progress("Initializing compressor...", 10)

            # Create compressor instance
            use_color = (color_mode == "Color")
            compressor = PredictiveCoding2D(num_bits=num_bits, use_color=use_color)

            # Run compression
            self.update_progress("Running compression algorithm...", 30)
            success = compressor.compress(image_path)

            if not success:
                raise Exception("Compression failed")

            # Save binary file
            self.update_progress("Saving compressed binary file...", 70)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            binary_file = os.path.join(output_folder, f"{base_name}_compressed.bin")
            compressor.save_binary_file(binary_file)

            # Save images
            self.update_progress("Saving visual images...", 90)
            compressor.save_images(output_folder)

            # Calculate compression stats
            original_size = os.path.getsize(image_path) * 8  # Convert bytes to bits
            compressed_size = os.path.getsize(binary_file) * 8

            # Get image data for display
            images = {
                'original': compressor.original,
                'predicted': compressor.predicted,
                'error': compressor.error,
                'quantized': compressor.quantized,
                'dequantized': compressor.dequantized,
                'decompressed': compressor.decompressed
            }

            # Create quantization info for results tab
            quant_info = []
            for i, q_info in enumerate(compressor.quantizers_info):
                quant_info.append({
                    'channel': i,
                    'min': float(q_info['min']),
                    'max': float(q_info['max'])
                })

            # Store compressor for later use
            self.compressor = compressor

            result = {
                'success': True,
                'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'images': images,
                'quantization_info': quant_info,
                'num_bits': num_bits,
                'image_folder': output_folder,
                'binary_file': binary_file,
                'color_mode': color_mode
            }

        except Exception as e:
            result = {'success': False, 'error': str(e)}

        # Update GUI in main thread
        self.root.after(0, self.on_compression_complete, result)

    def on_compression_complete(self, result):
        """Handle compression completion"""
        self.is_compressing = False
        self.compress_btn.config(state='normal')

        if result['success']:
            self.last_compression_result = result

            # Update console
            self.log_to_console(f"✓ Compression completed successfully!")
            self.log_to_console(f"  Compression Ratio: {result['compression_ratio']:.2f}:1")
            self.log_to_console(f"  Original Size: {result['original_size']} bits")
            self.log_to_console(f"  Compressed Size: {result['compressed_size']} bits")
            self.log_to_console(f"  Images saved to: {result['image_folder']}")
            self.log_to_console(f"  Binary file: {result['binary_file']}")

            # Update images display
            self.update_image_display(result['images'])

            # Update results tab
            self.update_results_tab(result)

            # Show success message
            messagebox.showinfo("Success", "Compression completed successfully!")

            # Switch to images tab
            self.notebook.select(self.images_tab)
        else:
            self.log_to_console(f"✗ Compression failed: {result['error']}")
            messagebox.showerror("Error", f"Compression failed: {result['error']}")

    def start_decompression_thread(self):
        """Start decompression in a separate thread"""
        if self.is_decompressing:
            messagebox.showwarning("Warning", "Decompression is already in progress!")
            return

        # Get inputs
        binary_file = self.binary_path_var.get()
        output_folder = self.decompress_output_var.get()

        # Validate inputs
        if not binary_file:
            messagebox.showerror("Error", "Please select a binary file!")
            return

        if not os.path.exists(binary_file):
            messagebox.showerror("Error", f"Binary file not found: {binary_file}")
            return

        # Create output folder if it doesn't exist
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Disable button and update status
        self.decompress_btn.config(state='disabled')
        self.is_decompressing = True
        self.update_progress("Starting decompression...", 0)

        # Start decompression thread
        self.decompression_thread = threading.Thread(
            target=self.run_decompression,
            args=(binary_file, output_folder)
        )
        self.decompression_thread.daemon = True
        self.decompression_thread.start()

    def run_decompression(self, binary_file, output_folder):
        """Run decompression in thread"""
        try:
            self.update_progress("Reading binary file...", 20)

            # Read binary file header
            with open(binary_file, 'rb') as f:
                # Read signature
                signature = f.read(4)
                if signature != b'PC2D':
                    raise ValueError("Invalid binary file format")

                # Read metadata
                width, height, channels, num_bits = struct.unpack('>HHBB', f.read(6))

                # Read quantizer ranges
                quantizers_info = []
                for _ in range(channels):
                    min_val, max_val = struct.unpack('>ff', f.read(8))
                    quantizers_info.append({'min': min_val, 'max': max_val})

                # Read compressed data
                compressed_data = f.read()

            self.update_progress("Reconstructing image...", 50)

            # Calculate total pixels
            total_pixels = width * height * channels

            # Unpack bits
            num_levels = 2 ** num_bits
            indices = []

            bit_buffer = 0
            bits_in_buffer = 0
            byte_index = 0

            for _ in range(total_pixels):
                while bits_in_buffer < num_bits:
                    if byte_index >= len(compressed_data):
                        break
                    bit_buffer = (bit_buffer << 8) | compressed_data[byte_index]
                    byte_index += 1
                    bits_in_buffer += 8

                shift = bits_in_buffer - num_bits
                level = (bit_buffer >> shift) & ((1 << num_bits) - 1)
                indices.append(level)

                bit_buffer &= (1 << shift) - 1
                bits_in_buffer -= num_bits

            # Reshape indices
            indices_array = np.array(indices, dtype=int).reshape(height, width, channels)

            # Create quantizer tables
            quantizer_tables = []
            for q_info in quantizers_info:
                min_val = q_info['min']
                max_val = q_info['max']
                step = (max_val - min_val) / num_levels
                quantizer = []
                for i in range(num_levels):
                    lower = min_val + i * step
                    upper = min_val + (i + 1) * step
                    dequant = (lower + upper) / 2.0
                    quantizer.append({
                        'level': i,
                        'lower': lower,
                        'upper': upper,
                        'dequant': dequant
                    })
                quantizer_tables.append(quantizer)

            self.update_progress("Applying predictive coding...", 70)

            # Reconstruct image using predictive coding
            decompressed = np.zeros((height, width, channels), dtype=float)

            for c in range(channels):
                channel_recon = np.zeros((height, width))
                quantizer = quantizer_tables[c]

                for y in range(height):
                    for x in range(width):
                        # Predict
                        if x == 0 and y == 0:
                            pred = 128.0
                        elif x == 0:
                            pred = channel_recon[y-1, x]
                        elif y == 0:
                            pred = channel_recon[y, x-1]
                        else:
                            # Adaptive predictor
                            A = channel_recon[y, x-1]
                            B = channel_recon[y-1, x]
                            C = channel_recon[y-1, x-1]

                            min_AC = min(A, C)
                            max_AC = max(A, C)

                            if B <= min_AC:
                                pred = max_AC
                            elif B >= max_AC:
                                pred = min_AC
                            else:
                                pred = A + C - B

                        # Get quantized error
                        level = indices_array[y, x, c]
                        deq_err = quantizer[level]['dequant']

                        # Reconstruct
                        recon_val = np.clip(pred + deq_err, 0, 255)
                        channel_recon[y, x] = recon_val
                        decompressed[y, x, c] = recon_val

            # If grayscale, remove channel dimension
            if channels == 1:
                decompressed = decompressed.squeeze()

            self.update_progress("Saving decompressed images...", 90)

            # Save decompressed image
            base_name = os.path.splitext(os.path.basename(binary_file))[0]
            decompressed_path = os.path.join(output_folder, f"{base_name}_decompressed.png")

            # Convert to uint8 for saving
            decompressed_uint8 = np.clip(decompressed, 0, 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV if color
            if channels == 3:
                decompressed_uint8 = cv2.cvtColor(decompressed_uint8, cv2.COLOR_RGB2BGR)

            cv2.imwrite(decompressed_path, decompressed_uint8)

            # Create simple images for display (using available data)
            images = {
                'decompressed': decompressed,
                'quantized': indices_array.astype(float),
                'original': np.zeros_like(decompressed),  # Not available in decompression
                'predicted': np.zeros_like(decompressed),  # Not available
                'error': np.zeros_like(decompressed),  # Not available
                'dequantized': np.zeros_like(decompressed)  # Not available
            }

            result = {
                'success': True,
                'images': images,
                'quantization_info': quantizers_info,
                'num_bits': num_bits,
                'image_folder': output_folder,
                'decompressed_file': decompressed_path,
                'width': width,
                'height': height,
                'channels': channels
            }

        except Exception as e:
            result = {'success': False, 'error': str(e)}

        # Update GUI in main thread
        self.root.after(0, self.on_decompression_complete, result)

    def on_decompression_complete(self, result):
        """Handle decompression completion"""
        self.is_decompressing = False
        self.decompress_btn.config(state='normal')

        if result['success']:
            # Update console
            self.log_to_console(f"✓ Decompression completed successfully!")
            self.log_to_console(f"  Image dimensions: {result['width']}x{result['height']}, {result['channels']} channel(s)")
            self.log_to_console(f"  Bits per pixel: {result['num_bits']}")
            self.log_to_console(f"  Decompressed image saved to: {result['decompressed_file']}")

            # Update images display
            self.update_image_display(result['images'])

            # Update results tab
            self.update_results_tab(result, is_decompression=True)

            # Show success message
            messagebox.showinfo("Success", "Decompression completed successfully!")

            # Switch to images tab
            self.notebook.select(self.images_tab)
        else:
            self.log_to_console(f"✗ Decompression failed: {result['error']}")
            messagebox.showerror("Error", f"Decompression failed: {result['error']}")

    def update_progress(self, message, value=None):
        """Update progress bar and status"""
        def update():
            self.status_label.config(text=message)
            if value is not None:
                self.progress_var.set(value)
            self.log_to_console(message)

        self.root.after(0, update)

    def log_to_console(self, message):
        """Add message to console"""
        def update():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.console_text.see(tk.END)

        self.root.after(0, update)

    def update_image_display(self, images):
        """Update the image display tab with new images"""
        def update():
            self.current_images = images

            # Display each image
            image_keys = ['original', 'predicted', 'error', 'quantized', 'dequantized', 'decompressed']

            for i, key in enumerate(image_keys):
                if key in images:
                    img = images[key]

                    # Skip if image is all zeros (not available in decompression)
                    if np.all(img == 0) and key not in ['quantized', 'decompressed']:
                        self.image_labels[i].config(image="", text="Not available")
                        continue

                    # Convert to displayable format
                    if key == 'quantized':
                        # Scale quantized indices for display
                        if img.max() > img.min():
                            display_img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        else:
                            display_img = img.astype(np.uint8)
                    else:
                        # Normalize other images
                        img_float = img.astype(float)
                        if img_float.max() > img_float.min():
                            display_img = ((img_float - img_float.min()) /
                                         (img_float.max() - img_float.min()) * 255).astype(np.uint8)
                        else:
                            display_img = img_float.astype(np.uint8)

                    # Convert to PhotoImage
                    if len(display_img.shape) == 3 and display_img.shape[2] == 3:
                        # RGB image
                        pil_image = Image.fromarray(display_img, 'RGB')
                    elif len(display_img.shape) == 3 and display_img.shape[2] == 1:
                        # Single channel 3D array
                        pil_image = Image.fromarray(display_img[:,:,0], 'L')
                    elif len(display_img.shape) == 2:
                        # Grayscale image
                        pil_image = Image.fromarray(display_img, 'L')
                    else:
                        pil_image = Image.fromarray(display_img)

                    # Resize for display
                    max_size = 300
                    width, height = pil_image.size
                    if width > max_size or height > max_size:
                        ratio = min(max_size/width, max_size/height)
                        new_size = (int(width * ratio), int(height * ratio))
                        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                    photo = ImageTk.PhotoImage(pil_image)

                    # Update label
                    self.image_labels[i].config(image=photo, text="")
                    self.image_labels[i].image = photo  # Keep reference
                else:
                    self.image_labels[i].config(image="", text="Image not available")

        self.root.after(0, update)

    def update_results_tab(self, result, is_decompression=False):
        """Update the results tab with compression/decompression info"""
        def update():
            text = ""

            if not is_decompression and 'quantization_info' in result:
                # Compression results
                text += "=" * 70 + "\n"
                text += "COMPRESSION RESULTS\n"
                text += "=" * 70 + "\n\n"

                text += f"Number of Bits: {result['num_bits']}\n"
                text += f"Color Mode: {result['color_mode']}\n"
                text += f"Compression Ratio: {result['compression_ratio']:.2f}:1\n"
                text += f"Original Size: {result['original_size']:,} bits\n"
                text += f"Compressed Size: {result['compressed_size']:,} bits\n\n"

                text += "QUANTIZATION INFORMATION\n"
                text += "-" * 40 + "\n"

                for i, q_info in enumerate(result['quantization_info']):
                    text += f"Channel {i+1}:\n"
                    text += f"  Min Error: {q_info['min']:.2f}\n"
                    text += f"  Max Error: {q_info['max']:.2f}\n"
                    text += f"  Range: {q_info['max'] - q_info['min']:.2f}\n"
                    text += f"  Step Size: {(q_info['max'] - q_info['min']) / (2**result['num_bits']):.2f}\n\n"

            elif is_decompression:
                # Decompression results
                text += "=" * 70 + "\n"
                text += "DECOMPRESSION RESULTS\n"
                text += "=" * 70 + "\n\n"

                text += f"Image Dimensions: {result['width']} x {result['height']}\n"
                text += f"Channels: {result['channels']}\n"
                text += f"Bits per pixel: {result['num_bits']}\n"
                text += f"Decompressed file: {result['decompressed_file']}\n\n"

                if 'quantization_info' in result:
                    text += "QUANTIZATION RANGES (from binary file)\n"
                    text += "-" * 40 + "\n"
                    for i, q_info in enumerate(result['quantization_info']):
                        text += f"Channel {i+1}: Min={q_info['min']:.2f}, Max={q_info['max']:.2f}\n"

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, text)

        self.root.after(0, update)


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main function"""
    root = tk.Tk()

    # Create and run the GUI
    app = ImageCompressionGUI(root)

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()


if __name__ == "__main__":
    main()
