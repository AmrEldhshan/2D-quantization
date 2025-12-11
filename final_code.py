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
            raise FileNotFoundError("Image path does not exist.")
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
                    if B < min(A, C):
                        P = max(A, C)
                    elif B > max(A, C):
                        P = min(A, C)
                    else:
                        P = A + C - B

                    self.predicted[y, x, ch] = P

        self.error = self.original - self.predicted
        return self.predicted, self.error

    # --------------------------
    # Uniform quantization
    # --------------------------
    def quantize_dequantize(self, error):
        levels = 2 ** self.num_bits
        e_min, e_max = error.min(), error.max()
        step = (e_max - e_min) / levels
        if step == 0:
            step = 1e-6

        quantized = np.floor((error - e_min) / step).astype(np.int32)
        quantized = np.clip(quantized, 0, levels - 1)
        dequantized = e_min + (quantized + 0.5) * step

        self.quantization_params = {
            'e_min': e_min,
            'e_max': e_max,
            'step': step,
            'levels': levels,
            'num_bits': self.num_bits
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
        # Scale quantized values to 0-255 for visualization
        if quantized.max() > quantized.min():
            # Normalize to 0-255
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
        """Save quantized data and parameters as .npz file"""
        np.savez_compressed(
            output_path,
            quantized=quantized,
            e_min=quant_params['e_min'],
            e_max=quant_params['e_max'],
            step=quant_params['step'],
            levels=quant_params['levels'],
            num_bits=quant_params['num_bits']
        )
        print(f"Quantized data (npz) saved to: {output_path}")

    # --------------------------
    # Load quantized from npz
    # --------------------------
    def load_quantized_npz(self, quantized_path):
        """Load quantized data from .npz file"""
        data = np.load(quantized_path)
        quantized = data['quantized']
        quant_params = {
            'e_min': data['e_min'].item(),
            'e_max': data['e_max'].item(),
            'step': data['step'].item(),
            'levels': data['levels'].item(),
            'num_bits': data['num_bits'].item()
        }
        return quantized, quant_params

    # --------------------------
    # Dequantize
    # --------------------------
    def dequantize(self, quantized, quant_params):
        """Dequantize using saved parameters"""
        e_min = quant_params['e_min']
        step = quant_params['step']
        dequantized = e_min + (quantized + 0.5) * step
        return dequantized

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
        self.save_quantized_npz(quantized, self.quantization_params, quantized_npz_path)

        # Save dequantized
        dequantized_path = os.path.join(folder, "5_dequantized.png")
        self.save_vis(dequantized, dequantized_path)

        # Create and save decompressed image
        # decompressed = predicted + dequantized
        # self.save_vis(decompressed, os.path.join(folder, "6_decompressed.png"))
        self.decompress("./output/2_predicted.png", "./output/5_dequantized.png", "6_decompressed.png")

        print("\n" + "="*60)
        print("COMPRESSION COMPLETE! Files saved in 'output' folder:")
        print("="*60)
        print("1. Original image:              1_original.png")
        print("2. Predicted image:             2_predicted.png")
        print("3. Error (with +128 offset):    3_error.png")
        print("4. Quantized (IMAGE version):   4_quantized.png")
        print("5. Quantized (NPZ version):     4_quantized.npz")
        print("6. Dequantized error:           5_dequantized.png")
        print("7. Final decompressed:          6_decompressed.png")
        print("="*60)

    # --------------------------
    # Decompress from quantized npz file
    # --------------------------
    def decompress_from_npz(self, predicted_path, quantized_npz_path, output_name="decompressed_from_npz.png"):
        """Decompress using predicted image and quantized npz file"""
        # Load predicted image
        predicted = self.load_image(predicted_path)

        # Load quantized data from npz
        quantized, quant_params = self.load_quantized_npz(quantized_npz_path)

        # Dequantize
        dequantized = self.dequantize(quantized, quant_params)

        # Reconstruct
        decompressed = predicted + dequantized

        # Save
        folder = "output"
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, output_name)
        self.save_vis(decompressed, output_path)

        print(f"\nDecompressed from npz file saved as: {output_path}")
        return decompressed

    # --------------------------
    # Decompress from images (original method)
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
# CLI class
# --------------------------
class CLI:
    def run(self):
        while True:
            print("\n" + "="*50)
            print("PREDICTIVE CODING COMPRESSION SYSTEM")
            print("="*50)
            print("1. Compress image (saves quantized as both PNG and NPZ)")
            print("2. Decompress from predicted + quantized NPZ")
            print("3. Decompress from predicted + dequantized images")
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

                try:
                    num_bits = int(input("Enter number of bits for quantization (default=2): ").strip() or 2)
                    if num_bits < 1 or num_bits > 8:
                        print("Warning: Using 2 bits (recommended 1-8 bits)")
                        num_bits = 2
                except ValueError:
                    num_bits = 2

                pc = PredictiveCoding2D(num_bits=num_bits)
                pc.compress(img_path)

            elif choice == "2":
                print("\n--- Decompress from NPZ file ---")
                predicted_path = input("Enter predicted image path: ").strip()
                quantized_npz_path = input("Enter quantized NPZ file path: ").strip()

                if not os.path.exists(predicted_path) or not os.path.exists(quantized_npz_path):
                    print("Error: One or both files not found!")
                    continue

                output_name = input("Enter output filename (default=decompressed_from_npz.png): ").strip()
                output_name = output_name if output_name else "decompressed_from_npz.png"

                pc = PredictiveCoding2D()
                pc.decompress_from_npz(predicted_path, quantized_npz_path, output_name)

            elif choice == "3":
                print("\n--- Decompress from images ---")
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

class PredictiveCodingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Predictive Coding")
        self.root.geometry("800x600")

        # Initialize PredictiveCoding2D
        self.pc = PredictiveCoding2D()

        # Variables
        self.selected_image = tk.StringVar()
        self.bits_var = tk.IntVar(value=2)
        self.predicted_path = tk.StringVar()
        self.dequantized_path = tk.StringVar()

        self.setup_ui()

    def setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Compress Tab
        self.compress_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.compress_frame, text="Compress")
        self.setup_compress_tab()

        # Decompress Tab
        self.decompress_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.decompress_frame, text="Decompress")
        self.setup_decompress_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_compress_tab(self):
        # Title
        title_label = ttk.Label(self.compress_frame, text="Image Compression", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Image selection
        frame1 = ttk.LabelFrame(self.compress_frame, text="Select Image", padding=10)
        frame1.pack(fill='x', padx=20, pady=5)

        self.image_entry = ttk.Entry(frame1, textvariable=self.selected_image, width=50)
        self.image_entry.pack(side=tk.LEFT, padx=(0, 10))

        browse_btn = ttk.Button(frame1, text="Browse", command=self.browse_image)
        browse_btn.pack(side=tk.LEFT)

        # Bits selection
        frame2 = ttk.LabelFrame(self.compress_frame, text="Quantization Bits", padding=10)
        frame2.pack(fill='x', padx=20, pady=5)

        bits_frame = ttk.Frame(frame2)
        bits_frame.pack()

        ttk.Label(bits_frame, text="Number of bits:").pack(side=tk.LEFT, padx=5)

        self.bits_entry = ttk.Entry(bits_frame, textvariable=self.bits_var, width=10)
        self.bits_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(bits_frame, text="(Enter a positive integer)",
                 font=('Arial', 9, 'italic')).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.compress_frame, length=400, mode='determinate')
        self.progress.pack(pady=10)

        self.progress_label = ttk.Label(self.compress_frame, text="")
        self.progress_label.pack()

        # Compress button
        compress_btn = ttk.Button(self.compress_frame, text="Compress Image",
                                  command=self.compress_image, style="Accent.TButton")
        compress_btn.pack(pady=20)

        # Open output folder button
        open_btn = ttk.Button(self.compress_frame, text="Open Output Folder",
                              command=self.open_output_folder)
        open_btn.pack()

    def setup_decompress_tab(self):
        # Title
        title_label = ttk.Label(self.decompress_frame, text="Image Decompression", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Predicted image selection
        frame1 = ttk.LabelFrame(self.decompress_frame, text="Predicted Image", padding=10)
        frame1.pack(fill='x', padx=20, pady=5)

        self.predicted_entry = ttk.Entry(frame1, textvariable=self.predicted_path, width=50)
        self.predicted_entry.pack(side=tk.LEFT, padx=(0, 10))

        browse_predicted_btn = ttk.Button(frame1, text="Browse",
                                          command=lambda: self.browse_file(self.predicted_path))
        browse_predicted_btn.pack(side=tk.LEFT)

        # Dequantized image selection
        frame2 = ttk.LabelFrame(self.decompress_frame, text="Dequantized Image", padding=10)
        frame2.pack(fill='x', padx=20, pady=5)

        self.dequantized_entry = ttk.Entry(frame2, textvariable=self.dequantized_path, width=50)
        self.dequantized_entry.pack(side=tk.LEFT, padx=(0, 10))

        browse_dequantized_btn = ttk.Button(frame2, text="Browse",
                                            command=lambda: self.browse_file(self.dequantized_path))
        browse_dequantized_btn.pack(side=tk.LEFT)

        # Progress bar
        self.decompress_progress = ttk.Progressbar(self.decompress_frame, length=400, mode='determinate')
        self.decompress_progress.pack(pady=10)

        self.decompress_label = ttk.Label(self.decompress_frame, text="")
        self.decompress_label.pack()

        # Decompress button
        decompress_btn = ttk.Button(self.decompress_frame, text="Decompress Image",
                                    command=self.decompress_image, style="Accent.TButton")
        decompress_btn.pack(pady=20)

        # Open output folder button
        open_btn = ttk.Button(self.decompress_frame, text="Open Output Folder",
                              command=self.open_output_folder)
        open_btn.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.selected_image.set(file_path)

    def browse_file(self, variable):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            variable.set(file_path)

    def update_progress(self, value, message):
        self.progress['value'] = value
        self.progress_label.config(text=message)
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_decompress_progress(self, value, message):
        self.decompress_progress['value'] = value
        self.decompress_label.config(text=message)
        self.status_var.set(message)
        self.root.update_idletasks()

    def compress_image(self):
        image_path = self.selected_image.get()

        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file.")
            return

        # Validate bits input
        try:
            num_bits = int(self.bits_var.get())
            if num_bits < 1:
                messagebox.showerror("Error", "Number of bits must be at least 1.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for number of bits.")
            return

        try:
            # Update PredictiveCoding2D with selected bits
            self.pc = PredictiveCoding2D(num_bits=num_bits)

            # Reset progress
            self.progress['value'] = 0
            self.progress_label.config(text="")

            # Start compression
            self.pc.compress(image_path)

            self.progress['value'] = 100
            self.progress_label.config(text="Compression complete!")
            self.status_var.set("Compression complete!")

            messagebox.showinfo("Success", "Compression completed successfully!\nOutput saved in 'output' folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Compression failed: {str(e)}")

    def decompress_image(self):
        predicted_path = self.predicted_path.get()
        dequantized_path = self.dequantized_path.get()

        if not predicted_path or not os.path.exists(predicted_path):
            messagebox.showerror("Error", "Please select a valid predicted image file.")
            return

        if not dequantized_path or not os.path.exists(dequantized_path):
            messagebox.showerror("Error", "Please select a valid dequantized image file.")
            return

        try:
            # Reset progress
            self.decompress_progress['value'] = 0
            self.decompress_label.config(text="")

            # Start decompression
            self.pc.decompress(predicted_path, dequantized_path, "decompressed_only")

            self.decompress_progress['value'] = 100
            self.decompress_label.config(text="Decompression complete!")
            self.status_var.set("Decompression complete!")

            messagebox.showinfo("Success",
                               f"Decompression completed successfully!\nOutput saved as: output/decompressed_only.png")
        except Exception as e:
            messagebox.showerror("Error", f"Decompression failed: {str(e)}")

    def open_output_folder(self):
        output_folder = "output"
        if os.path.exists(output_folder):
            os.startfile(output_folder) if os.name == 'nt' else os.system(f'open "{output_folder}"')
        else:
            messagebox.showinfo("Info", "Output folder doesn't exist yet. Run compression first.")


# --------------------------
# Main Application
# --------------------------
def main():
    root = tk.Tk()

    # Set ttk theme
    style = ttk.Style()
    style.theme_use('clam')

    # Create custom style for accent button
    style.configure("Accent.TButton", font=('Arial', 10, 'bold'))

    app = PredictiveCodingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# # --------------------------
# # Main
# # --------------------------
# if __name__ == "__main__":
    # CLI().run()
