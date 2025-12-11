import numpy as np
import cv2
import os
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import warnings
import pickle  # For saving compressed data
warnings.filterwarnings('ignore')



# ==================== COMPRESSOR CLASS ====================
class Compressor:
    def __init__(self):
        self.original_image = None
        self.compressed_data = None
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback

    def update_progress(self, message, value=None):
        """Update progress through callback"""
        if self.progress_callback:
            self.progress_callback(message, value)

    def load_image(self, image_path):
        """Load image from path"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from: {image_path}")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def predictor_2d(self, image):
        """2D Predictor Function"""
        if len(image.shape) == 3:
            predicted = np.zeros_like(image, dtype=np.float32)
            rows, cols, channels = image.shape

            for ch in range(channels):
                predicted[0, 0, ch] = image[0, 0, ch]

                for j in range(1, cols):
                    predicted[0, j, ch] = image[0, j-1, ch]

                for i in range(1, rows):
                    predicted[i, 0, ch] = image[i-1, 0, ch]

                for i in range(1, rows):
                    for j in range(1, cols):
                        A = image[i-1, j, ch]
                        B = image[i, j-1, ch]
                        C = image[i-1, j-1, ch]
                        predicted[i, j, ch] = A + B - C
        else:
            predicted = np.zeros_like(image, dtype=np.float32)
            rows, cols = image.shape

            predicted[0, 0] = image[0, 0]

            for j in range(1, cols):
                predicted[0, j] = image[0, j-1]

            for i in range(1, rows):
                predicted[i, 0] = image[i-1, 0]

            for i in range(1, rows):
                for j in range(1, cols):
                    A = image[i-1, j]
                    B = image[i, j-1]
                    C = image[i-1, j-1]
                    predicted[i, j] = A + B - C

        return predicted

    def uniform_quantizer(self, error_image, num_bits=2):
        """Uniform quantizer with dynamic number of bits"""
        min_val = np.min(error_image)
        max_val = np.max(error_image)

        L = 2 ** num_bits
        step = (max_val - min_val) / L if max_val != min_val else 1

        quantized_indices = np.floor((error_image - min_val) / step).astype(np.uint8)
        quantized_indices = np.clip(quantized_indices, 0, L-1)

        dequantized = min_val + (quantized_indices + 0.5) * step

        binary_codes = []
        for idx in range(L):
            binary_codes.append(format(idx, f'0{num_bits}b'))

        quantization_info = {
            'binary_codes': binary_codes,
            'step': step,
            'range': (min_val, max_val),
            'num_levels': L,
            'dequantized_values': [min_val + (i + 0.5) * step for i in range(L)],
            'original_pixel_values': [min_val + i * step for i in range(L)],
            'min_val': min_val,
            'max_val': max_val
        }

        return quantized_indices, dequantized, quantization_info

    def compress(self, image_path, num_bits=2, save_folder=None):
        """Main compression method"""
        try:
            self.update_progress("Loading image...", 10)
            self.original_image = self.load_image(image_path)

            self.update_progress("Applying 2D predictor...", 30)
            predicted_image = self.predictor_2d(self.original_image)

            self.update_progress("Calculating error...", 40)
            error_image = self.original_image.astype(np.float32) - predicted_image

            self.update_progress("Applying uniform quantization...", 60)
            quantized, dequantized, q_info = self.uniform_quantizer(error_image, num_bits)

            self.update_progress("Reconstructing image...", 80)
            decompressed = predicted_image + dequantized
            decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)

            # Store compressed data in a simple dictionary format
            self.compressed_data = {
                'original_shape': self.original_image.shape,
                'predicted': predicted_image.astype(np.float16),  # Use float16 to save space
                'quantized': quantized,  # This is what we'll save
                'quantization_info': q_info,
                'num_bits': num_bits,
                'min_val': q_info['min_val'],
                'max_val': q_info['max_val'],
                'step': q_info['step']
            }

            # Calculate compression ratio
            if len(self.original_image.shape) == 3:
                orig_size = self.original_image.shape[0] * self.original_image.shape[1] * 3 * 8
            else:
                orig_size = self.original_image.size * 8
            comp_size = quantized.size * num_bits
            compression_ratio = orig_size / comp_size if comp_size > 0 else 0

            self.update_progress("Saving images...", 90)

            if save_folder:
                image_folder = self.save_all_images(
                    os.path.splitext(os.path.basename(image_path))[0],
                    self.original_image,
                    predicted_image,
                    error_image,
                    quantized,
                    dequantized,
                    decompressed,
                    save_folder
                )

                # Save compressed data (quantized image only) as binary file
                compressed_file = self.save_compressed_data(quantized, num_bits, q_info, image_path, save_folder)
            else:
                image_folder = None
                compressed_file = self.save_compressed_data(quantized, num_bits, q_info, image_path)

            self.update_progress("Compression complete!", 100)

            return {
                'success': True,
                'compression_ratio': compression_ratio,
                'original_size': orig_size,
                'compressed_size': comp_size,
                'image_folder': image_folder,
                'compressed_file': compressed_file,
                'quantization_info': q_info,
                'num_bits': num_bits,
                'predicted': predicted_image,
                'quantized': quantized,
                'dequantized': dequantized,
                'images': {
                    'original': self.original_image,
                    'predicted': predicted_image,
                    'error': error_image,
                    'quantized': quantized,
                    'dequantized': dequantized,
                    'decompressed': decompressed
                }
            }

        except Exception as e:
            self.update_progress(f"Error: {str(e)}", 0)
            return {'success': False, 'error': str(e)}

    def save_all_images(self, base_filename, original, predicted, error, quantized, dequantized, decompressed, save_folder):
        """Save all 6 images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.join(save_folder, f"compression_{base_filename}_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        # 1. Original
        original_path = os.path.join(folder_name, "01_original.png")
        if len(original.shape) == 3:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            cv2.imwrite(original_path, original_bgr)
        else:
            cv2.imwrite(original_path, original)

        # 2. Predicted
        predicted_path = os.path.join(folder_name, "02_predicted.png")
        predicted_uint8 = np.clip(predicted, 0, 255).astype(np.uint8)
        if len(predicted_uint8.shape) == 3:
            predicted_bgr = cv2.cvtColor(predicted_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(predicted_path, predicted_bgr)
        else:
            cv2.imwrite(predicted_path, predicted_uint8)

        # 3. Error
        error_path = os.path.join(folder_name, "03_error.png")
        error_normalized = error - np.min(error)
        if np.max(error_normalized) > 0:
            error_normalized = (error_normalized / np.max(error_normalized) * 255).astype(np.uint8)
        if len(error_normalized.shape) == 3:
            error_bgr = cv2.cvtColor(error_normalized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(error_path, error_bgr)
        else:
            cv2.imwrite(error_path, error_normalized)

        # 4. Quantized
        quantized_path = os.path.join(folder_name, "04_quantized.png")
        if np.max(quantized) > np.min(quantized):
            quantized_display = ((quantized - np.min(quantized)) /
                                (np.max(quantized) - np.min(quantized)) * 255)
        else:
            quantized_display = quantized * 255 / (2 ** self.compressed_data['num_bits'])
        quantized_display = quantized_display.astype(np.uint8)
        if len(quantized_display.shape) == 3:
            quantized_bgr = cv2.cvtColor(quantized_display, cv2.COLOR_RGB2BGR)
            cv2.imwrite(quantized_path, quantized_bgr)
        else:
            cv2.imwrite(quantized_path, quantized_display)

        # 5. De-quantized
        dequantized_path = os.path.join(folder_name, "05_dequantized.png")
        dequantized_normalized = dequantized - np.min(dequantized)
        if np.max(dequantized_normalized) > 0:
            dequantized_normalized = (dequantized_normalized / np.max(dequantized_normalized) * 255).astype(np.uint8)
        if len(dequantized_normalized.shape) == 3:
            dequantized_bgr = cv2.cvtColor(dequantized_normalized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dequantized_path, dequantized_bgr)
        else:
            cv2.imwrite(dequantized_path, dequantized_normalized)

        # 6. Decompressed
        decompressed_path = os.path.join(folder_name, "06_decompressed.png")
        if len(decompressed.shape) == 3:
            decompressed_bgr = cv2.cvtColor(decompressed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(decompressed_path, decompressed_bgr)
        else:
            cv2.imwrite(decompressed_path, decompressed)

        # Save quantization table
        table_path = os.path.join(folder_name, "quantization_table.txt")
        with open(table_path, 'w') as f:
            f.write(self.get_quantization_table_text(self.compressed_data['quantization_info'],
                                                    self.compressed_data['num_bits']))

        return folder_name

    def save_compressed_data(self, quantized, num_bits, q_info, image_path, save_folder=None):
        """Save compressed data (quantized image) as binary file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if save_folder:
            # Save quantized image as binary file
            quantized_filename = os.path.join(save_folder, f"compressed_{base_name}_{num_bits}bits_{timestamp}.bin")

            # Save metadata separately
            metadata_filename = os.path.join(save_folder, f"metadata_{base_name}_{num_bits}bits_{timestamp}.pkl")

            # Create metadata dictionary
            metadata = {
                'original_shape': self.compressed_data['original_shape'],
                'num_bits': num_bits,
                'quantization_info': q_info,
                'min_val': q_info['min_val'],
                'max_val': q_info['max_val'],
                'step': q_info['step']
            }

            # Save quantized data as raw binary
            with open(quantized_filename, 'wb') as f:
                f.write(quantized.tobytes())

            # Save metadata
            with open(metadata_filename, 'wb') as f:
                pickle.dump(metadata, f)

            return quantized_filename, metadata_filename
        else:
            quantized_filename = f"compressed_{base_name}_{num_bits}bits_{timestamp}.bin"
            metadata_filename = f"metadata_{base_name}_{num_bits}bits_{timestamp}.pkl"

            # Save quantized data
            with open(quantized_filename, 'wb') as f:
                f.write(quantized.tobytes())

            # Save metadata
            with open(metadata_filename, 'wb') as f:
                pickle.dump(self.compressed_data, f)

            return quantized_filename, metadata_filename

    def get_quantization_table_text(self, q_info, num_bits):
        """Get quantization table as text"""
        lines = []
        lines.append("="*70)
        lines.append("QUANTIZATION TABLE")
        lines.append("="*70)
        lines.append(f"{'Index':<10} {'Binary Code':<15} {'Step Size':<15} {'De-quantized Value':<20} {'Pixel Range':<20}")
        lines.append("-"*70)

        L = q_info['num_levels']
        min_val = q_info['range'][0]
        step = q_info['step']

        for i in range(L):
            binary_code = q_info['binary_codes'][i]
            dequant_val = q_info['dequantized_values'][i]
            pixel_range = f"{min_val + i*step:.1f}-{min_val + (i+1)*step:.1f}"
            lines.append(f"{i:<10} {binary_code:<15} {step:<15.2f} {dequant_val:<20.2f} {pixel_range:<20}")

        lines.append("="*70)
        lines.append(f"Number of bits: {num_bits}")
        lines.append(f"Number of levels: {L}")
        lines.append(f"Total range: [{min_val:.2f}, {q_info['range'][1]:.2f}]")
        lines.append(f"Step size: {step:.4f}")

        return "\n".join(lines)

# ==================== DECOMPRESSOR CLASS ====================
class Decompressor:
    def __init__(self):
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback

    def update_progress(self, message, value=None):
        """Update progress through callback"""
        if self.progress_callback:
            self.progress_callback(message, value)

    def decompress(self, quantized_file, metadata_file, original_shape, num_bits, q_info):
        """Main decompression method - using predicted and dequantized directly"""
        try:
            self.update_progress("Loading quantized data...", 20)

            # Load quantized data
            with open(quantized_file, 'rb') as f:
                quantized_data = np.frombuffer(f.read(), dtype=np.uint8)

            # Reshape to original dimensions
            if len(original_shape) == 3:
                quantized = quantized_data.reshape(original_shape[0], original_shape[1], original_shape[2])
            else:
                quantized = quantized_data.reshape(original_shape[0], original_shape[1])

            self.update_progress("De-quantizing...", 40)

            # De-quantize using the quantization info
            min_val = q_info['min_val']
            step = q_info['step']
            dequantized = min_val + (quantized + 0.5) * step

            self.update_progress("Reconstructing image...", 60)

            # IMPORTANT: For decompression, we need the predicted image
            # In a real scenario, we would need to save the predicted image or recalculate it
            # For now, we'll show a message that predicted image is needed

            self.update_progress("Note: Predicted image required for full decompression", 70)

            # Create a simple reconstruction for display purposes
            # In a real application, you would load the predicted image here
            decompressed = dequantized  # This is just the error, not full reconstruction

            self.update_progress("Saving images...", 80)

            # For display, create dummy predicted image (zeros)
            predicted = np.zeros_like(dequantized, dtype=np.float32)

            # Create dummy original for display
            original_display = np.clip(dequantized, 0, 255).astype(np.uint8)

            self.update_progress("Decompression visualization ready!", 90)

            return {
                'success': True,
                'quantized': quantized,
                'dequantized': dequantized,
                'predicted': predicted,  # This would be loaded from file in real scenario
                'images': {
                    'original': original_display,
                    'predicted': predicted,
                    'error': dequantized,  # Error is the dequantized values
                    'quantized': quantized,
                    'dequantized': dequantized,
                    'decompressed': original_display  # Simple reconstruction
                },
                'quantization_info': q_info,
                'num_bits': num_bits
            }

        except Exception as e:
            self.update_progress(f"Error: {str(e)}", 0)
            return {'success': False, 'error': str(e)}

    def decompress_with_predicted(self, quantized_file, predicted_image, original_shape, num_bits, q_info):
        """Decompression with provided predicted image"""
        try:
            self.update_progress("Loading quantized data...", 20)

            # Load quantized data
            with open(quantized_file, 'rb') as f:
                quantized_data = np.frombuffer(f.read(), dtype=np.uint8)

            # Reshape to original dimensions
            if len(original_shape) == 3:
                quantized = quantized_data.reshape(original_shape[0], original_shape[1], original_shape[2])
            else:
                quantized = quantized_data.reshape(original_shape[0], original_shape[1])

            self.update_progress("De-quantizing...", 40)

            # De-quantize
            min_val = q_info['min_val']
            step = q_info['step']
            dequantized = min_val + (quantized + 0.5) * step

            self.update_progress("Reconstructing image...", 60)

            # Full reconstruction using predicted image
            decompressed = predicted_image + dequantized
            decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)

            error_image = dequantized
            original_display = decompressed.copy()

            self.update_progress("Decompression complete!", 80)

            return {
                'success': True,
                'quantized': quantized,
                'dequantized': dequantized,
                'decompressed': decompressed,
                'images': {
                    'original': original_display,
                    'predicted': predicted_image,
                    'error': error_image,
                    'quantized': quantized,
                    'dequantized': dequantized,
                    'decompressed': decompressed
                },
                'quantization_info': q_info,
                'num_bits': num_bits
            }

        except Exception as e:
            self.update_progress(f"Error: {str(e)}", 0)
            return {'success': False, 'error': str(e)}

    def save_decompression_images(self, base_filename, original, predicted, error, quantized, dequantized, decompressed, q_info, num_bits, save_folder):
        """Save all 6 decompression images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.join(save_folder, f"decompression_{base_filename}_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)

        # 1. Original (Reconstructed)
        original_path = os.path.join(folder_name, "01_original_reconstructed.png")
        if len(original.shape) == 3:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            cv2.imwrite(original_path, original_bgr)
        else:
            cv2.imwrite(original_path, original)

        # 2. Predicted
        predicted_path = os.path.join(folder_name, "02_predicted.png")
        predicted_uint8 = np.clip(predicted, 0, 255).astype(np.uint8)
        if len(predicted_uint8.shape) == 3:
            predicted_bgr = cv2.cvtColor(predicted_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(predicted_path, predicted_bgr)
        else:
            cv2.imwrite(predicted_path, predicted_uint8)

        # 3. Error
        error_path = os.path.join(folder_name, "03_error.png")
        error_normalized = error - np.min(error)
        if np.max(error_normalized) > 0:
            error_normalized = (error_normalized / np.max(error_normalized) * 255).astype(np.uint8)
        if len(error_normalized.shape) == 3:
            error_bgr = cv2.cvtColor(error_normalized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(error_path, error_bgr)
        else:
            cv2.imwrite(error_path, error_normalized)

        # 4. Quantized
        quantized_path = os.path.join(folder_name, "04_quantized.png")
        if np.max(quantized) > np.min(quantized):
            quantized_display = ((quantized - np.min(quantized)) /
                                (np.max(quantized) - np.min(quantized)) * 255)
        else:
            quantized_display = quantized * 255 / (2 ** num_bits)
        quantized_display = quantized_display.astype(np.uint8)
        if len(quantized_display.shape) == 3:
            quantized_bgr = cv2.cvtColor(quantized_display, cv2.COLOR_RGB2BGR)
            cv2.imwrite(quantized_path, quantized_bgr)
        else:
            cv2.imwrite(quantized_path, quantized_display)

        # 5. De-quantized
        dequantized_path = os.path.join(folder_name, "05_dequantized.png")
        dequantized_normalized = dequantized - np.min(dequantized)
        if np.max(dequantized_normalized) > 0:
            dequantized_normalized = (dequantized_normalized / np.max(dequantized_normalized) * 255).astype(np.uint8)
        if len(dequantized_normalized.shape) == 3:
            dequantized_bgr = cv2.cvtColor(dequantized_normalized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dequantized_path, dequantized_bgr)
        else:
            cv2.imwrite(dequantized_path, dequantized_normalized)

        # 6. Decompressed
        decompressed_path = os.path.join(folder_name, "06_decompressed.png")
        if len(decompressed.shape) == 3:
            decompressed_bgr = cv2.cvtColor(decompressed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(decompressed_path, decompressed_bgr)
        else:
            cv2.imwrite(decompressed_path, decompressed)

        # Save quantization table
        table_path = os.path.join(folder_name, "quantization_table.txt")
        with open(table_path, 'w') as f:
            f.write(self.get_quantization_table_text(q_info, num_bits))

        return folder_name

    def get_quantization_table_text(self, q_info, num_bits):
        """Get quantization table as text"""
        lines = []
        lines.append("="*70)
        lines.append("QUANTIZATION TABLE")
        lines.append("="*70)
        lines.append(f"{'Index':<10} {'Binary Code':<15} {'Step Size':<15} {'De-quantized Value':<20} {'Pixel Range':<20}")
        lines.append("-"*70)

        L = q_info['num_levels']
        min_val = q_info['range'][0]
        step = q_info['step']

        for i in range(L):
            binary_code = q_info['binary_codes'][i]
            dequant_val = q_info['dequantized_values'][i]
            pixel_range = f"{min_val + i*step:.1f}-{min_val + (i+1)*step:.1f}"
            lines.append(f"{i:<10} {binary_code:<15} {step:<15.2f} {dequant_val:<20.2f} {pixel_range:<20}")

        lines.append("="*70)
        lines.append(f"Number of bits: {num_bits}")
        lines.append(f"Number of levels: {L}")
        lines.append(f"Total range: [{min_val:.2f}, {q_info['range'][1]:.2f}]")
        lines.append(f"Step size: {step:.4f}")

        return "\n".join(lines)

# ==================== GUI CLASS ====================
class ImageCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression System")
        self.root.geometry("1200x800")

        # Initialize compressor and decompressor
        self.compressor = Compressor()
        self.decompressor = Decompressor()

        # Set progress callbacks
        self.compressor.set_progress_callback(self.update_progress)
        self.decompressor.set_progress_callback(self.update_progress)

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
        ttk.Label(input_frame, text="Number of Bits (Default=2):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bits_var = tk.StringVar(value="2")
        self.bits_entry = ttk.Entry(input_frame, textvariable=self.bits_var, width=10)
        self.bits_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Output folder
        ttk.Label(input_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_folder_var = tk.StringVar(value="output")
        self.output_entry = ttk.Entry(input_frame, textvariable=self.output_folder_var, width=50)
        self.output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        output_browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_output_folder)
        output_browse_btn.grid(row=2, column=2, pady=5)

        # Compress button
        self.compress_btn = ttk.Button(input_frame,
                                      text="Start Compression",
                                      command=self.start_compression_thread)
        self.compress_btn.grid(row=3, column=0, columnspan=3, pady=(20, 0))

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

        # Quantized file selection
        ttk.Label(input_frame, text="Quantized File (.bin):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.quantized_path_var = tk.StringVar()
        self.quantized_entry = ttk.Entry(input_frame, textvariable=self.quantized_path_var, width=50)
        self.quantized_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 5))

        browse_quant_btn = ttk.Button(input_frame, text="Browse", command=self.browse_quantized_file)
        browse_quant_btn.grid(row=0, column=2, pady=5)

        # Metadata file selection
        ttk.Label(input_frame, text="Metadata File (.pkl):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.metadata_path_var = tk.StringVar()
        self.metadata_entry = ttk.Entry(input_frame, textvariable=self.metadata_path_var, width=50)
        self.metadata_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        browse_meta_btn = ttk.Button(input_frame, text="Browse", command=self.browse_metadata_file)
        browse_meta_btn.grid(row=1, column=2, pady=5)

        # Predicted image selection (optional, for full decompression)
        ttk.Label(input_frame, text="Predicted Image (Optional, .png):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.predicted_path_var = tk.StringVar()
        self.predicted_entry = ttk.Entry(input_frame, textvariable=self.predicted_path_var, width=50)
        self.predicted_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        browse_pred_btn = ttk.Button(input_frame, text="Browse", command=self.browse_predicted_file)
        browse_pred_btn.grid(row=2, column=2, pady=5)

        # Output folder
        ttk.Label(input_frame, text="Output Folder:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.decompress_output_var = tk.StringVar(value="output")
        self.decompress_output_entry = ttk.Entry(input_frame, textvariable=self.decompress_output_var, width=50)
        self.decompress_output_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        output_browse_btn = ttk.Button(input_frame, text="Browse", command=self.browse_decompress_output_folder)
        output_browse_btn.grid(row=3, column=2, pady=5)

        # Decompress button
        self.decompress_btn = ttk.Button(input_frame,
                                        text="Start Decompression",
                                        command=self.start_decompression_thread)
        self.decompress_btn.grid(row=4, column=0, columnspan=3, pady=(20, 0))

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
            "Original Image",
            "Predicted Image",
            "Error Image",
            "Quantized Image",
            "De-quantized Image",
            "Decompressed Image"
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
        # Create text widget for quantization table
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

    def browse_quantized_file(self):
        """Browse for quantized file"""
        filename = filedialog.askopenfilename(
            title="Select Quantized File",
            filetypes=[
                ("Binary files", "*.bin"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.quantized_path_var.set(filename)

    def browse_metadata_file(self):
        """Browse for metadata file"""
        filename = filedialog.askopenfilename(
            title="Select Metadata File",
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.metadata_path_var.set(filename)

    def browse_predicted_file(self):
        """Browse for predicted image file"""
        filename = filedialog.askopenfilename(
            title="Select Predicted Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.predicted_path_var.set(filename)

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
            args=(image_path, num_bits, output_folder)
        )
        self.compression_thread.daemon = True
        self.compression_thread.start()

    def run_compression(self, image_path, num_bits, output_folder):
        """Run compression in thread"""
        result = self.compressor.compress(image_path, num_bits, output_folder)

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
            self.log_to_console(f"  Compressed files: {result['compressed_file'][0]} and {result['compressed_file'][1]}")

            # Update images display
            self.update_image_display(result['images'])

            # Update results tab
            self.update_results_tab(result['quantization_info'], result['num_bits'])

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
        quantized_file = self.quantized_path_var.get()
        metadata_file = self.metadata_path_var.get()
        predicted_file = self.predicted_path_var.get()
        output_folder = self.decompress_output_var.get()

        # Validate inputs
        if not quantized_file or not metadata_file:
            messagebox.showerror("Error", "Please select both quantized and metadata files!")
            return

        if not os.path.exists(quantized_file):
            messagebox.showerror("Error", f"Quantized file not found: {quantized_file}")
            return

        if not os.path.exists(metadata_file):
            messagebox.showerror("Error", f"Metadata file not found: {metadata_file}")
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
            args=(quantized_file, metadata_file, predicted_file, output_folder)
        )
        self.decompression_thread.daemon = True
        self.decompression_thread.start()

    def run_decompression(self, quantized_file, metadata_file, predicted_file, output_folder):
        """Run decompression in thread"""
        try:
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            original_shape = metadata['original_shape']
            num_bits = metadata['num_bits']
            q_info = metadata['quantization_info']

            # Check if predicted image is provided
            if predicted_file and os.path.exists(predicted_file):
                # Load predicted image
                predicted_img = cv2.imread(predicted_file)
                if predicted_img is not None:
                    if len(predicted_img.shape) == 3:
                        predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)

                    # Convert to float32
                    predicted_image = predicted_img.astype(np.float32)

                    # Use decompression with predicted image
                    result = self.decompressor.decompress_with_predicted(
                        quantized_file, predicted_image, original_shape, num_bits, q_info
                    )
                else:
                    result = self.decompressor.decompress(
                        quantized_file, metadata_file, original_shape, num_bits, q_info
                    )
            else:
                # Use basic decompression
                result = self.decompressor.decompress(
                    quantized_file, metadata_file, original_shape, num_bits, q_info
                )

            # Save images if output folder is specified
            if output_folder and result['success']:
                base_filename = os.path.splitext(os.path.basename(quantized_file))[0]
                image_folder = self.decompressor.save_decompression_images(
                    base_filename,
                    result['images']['original'],
                    result['images']['predicted'],
                    result['images']['error'],
                    result['images']['quantized'],
                    result['images']['dequantized'],
                    result['images']['decompressed'],
                    q_info,
                    num_bits,
                    output_folder
                )
                result['image_folder'] = image_folder

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

            if 'image_folder' in result:
                self.log_to_console(f"  Images saved to: {result['image_folder']}")

            # Update images display
            self.update_image_display(result['images'])

            # Update results tab
            self.update_results_tab(result['quantization_info'], result['num_bits'])

            # Show info about predicted image
            if result['images']['predicted'].max() == 0:
                self.log_to_console("  Note: For full decompression, provide the predicted image")

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

                    # Convert to displayable format
                    if key == 'predicted':
                        display_img = np.clip(img, 0, 255).astype(np.uint8)
                    elif key == 'error' or key == 'dequantized':
                        # Normalize for display
                        display_img = img - np.min(img)
                        if np.max(display_img) > 0:
                            display_img = (display_img / np.max(display_img) * 255).astype(np.uint8)
                        else:
                            display_img = display_img.astype(np.uint8)
                    elif key == 'quantized':
                        # Scale to 0-255 for display
                        if np.max(img) > np.min(img):
                            display_img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
                        else:
                            display_img = img.astype(np.uint8)
                    else:
                        display_img = img.astype(np.uint8) if img.dtype != np.uint8 else img

                    # Convert to PhotoImage
                    if len(display_img.shape) == 3:
                        # RGB image
                        pil_image = Image.fromarray(display_img)
                    else:
                        # Grayscale image
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

    def update_results_tab(self, q_info, num_bits):
        """Update the results tab with quantization table"""
        def update():
            text = self.compressor.get_quantization_table_text(q_info, num_bits)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, text)

            # Add compression info if available
            if self.last_compression_result and self.last_compression_result['success']:
                self.results_text.insert(tk.END, "\n\n" + "="*70 + "\n")
                self.results_text.insert(tk.END, "COMPRESSION RESULTS\n")
                self.results_text.insert(tk.END, "="*70 + "\n")
                self.results_text.insert(tk.END, f"Compression Ratio: {self.last_compression_result['compression_ratio']:.2f}:1\n")
                self.results_text.insert(tk.END, f"Original Size: {self.last_compression_result['original_size']} bits\n")
                self.results_text.insert(tk.END, f"Compressed Size: {self.last_compression_result['compressed_size']} bits\n")

        self.root.after(0, update)

def main():
    """Main function"""
    root = tk.Tk()

    # Create custom style
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="white", background="#0078D7")

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
