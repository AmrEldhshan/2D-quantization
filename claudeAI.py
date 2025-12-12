import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os

class ImageCompressionSystem:
    def __init__(self):
        self.original_image = None
        self.predicted_image = None
        self.error_image = None
        self.quantized_image = None
        self.dequantized_image = None
        self.decompressed_image = None
        self.num_bits = 2
        self.binary_data = ""

    def predict_2d(self, img, i, j):
        """2D Predictor Function"""
        height, width = img.shape

        # First pixel
        if i == 0 and j == 0:
            return 128

        # First row
        if i == 0:
            return img[i, j-1]

        # First column
        if j == 0:
            return img[i-1, j]

        # Other pixels: a + b - c
        a = img[i, j-1]      # left
        b = img[i-1, j]      # top
        c = img[i-1, j-1]    # top-left

        return int(a + b - c)

    def uniform_quantize(self, value, num_bits):
        """Uniform Quantization"""
        levels = 2 ** num_bits
        min_val = -255
        max_val = 255
        range_val = max_val - min_val
        step = range_val / levels

        # Quantization
        quant_index = int((value - min_val) / step)
        quant_index = max(0, min(levels - 1, quant_index))

        # Binary code
        binary_code = format(quant_index, f'0{num_bits}b')

        # De-quantization value (reconstruction level)
        dequant_value = min_val + (quant_index + 0.5) * step

        # Range
        quant_range = [min_val + quant_index * step, min_val + (quant_index + 1) * step]

        return {
            'quant_index': quant_index,
            'binary_code': binary_code,
            'step': step,
            'range': quant_range,
            'dequant_value': int(dequant_value)
        }

    def compress_image(self, image_path, num_bits=2):
        """Compress Image"""
        print(f"\n{'='*60}")
        print(f"COMPRESSION PROCESS STARTED")
        print(f"{'='*60}")

        # Load image and convert to grayscale
        img = Image.open(image_path).convert('L')
        self.original_image = np.array(img, dtype=np.float64)
        self.num_bits = num_bits

        height, width = self.original_image.shape
        print(f"Image Size: {width}x{height}")
        print(f"Number of Bits: {num_bits}")

        # Initialize arrays
        self.predicted_image = np.zeros_like(self.original_image)
        self.error_image = np.zeros_like(self.original_image)
        self.quantized_image = np.zeros_like(self.original_image)
        self.dequantized_image = np.zeros_like(self.original_image)

        self.binary_data = ""

        # Process each pixel
        print("\nProcessing pixels...")
        for i in range(height):
            for j in range(width):
                # Prediction
                self.predicted_image[i, j] = self.predict_2d(self.original_image, i, j)

                # Error
                self.error_image[i, j] = self.original_image[i, j] - self.predicted_image[i, j]

                # Quantization
                quant_result = self.uniform_quantize(self.error_image[i, j], num_bits)
                self.quantized_image[i, j] = quant_result['quant_index']
                self.dequantized_image[i, j] = quant_result['dequant_value']
                self.binary_data += quant_result['binary_code']

        print("Compression completed!")

        # Save binary file
        self.save_binary_file()

        # Calculate compression ratio
        original_size = height * width * 8  # 8 bits per pixel
        compressed_size = height * width * num_bits
        compression_ratio = original_size / compressed_size

        print(f"\n{'='*60}")
        print(f"COMPRESSION STATISTICS")
        print(f"{'='*60}")
        print(f"Original Size: {original_size} bits ({original_size/8:.2f} bytes)")
        print(f"Compressed Size: {compressed_size} bits ({compressed_size/8:.2f} bytes)")
        print(f"Compression Ratio: {compression_ratio:.2f}:1")
        print(f"Space Saved: {((original_size - compressed_size) / original_size * 100):.2f}%")

    def decompress_image(self):
        """Decompress Image"""
        print(f"\n{'='*60}")
        print(f"DECOMPRESSION PROCESS STARTED")
        print(f"{'='*60}")

        height, width = self.quantized_image.shape
        self.decompressed_image = np.zeros_like(self.original_image)

        print("Reconstructing image...")
        for i in range(height):
            for j in range(width):
                # Predict using already decompressed pixels
                predicted_val = self.predict_2d(self.decompressed_image, i, j)

                # Add dequantized error
                self.decompressed_image[i, j] = predicted_val + self.dequantized_image[i, j]

                # Clip to valid range
                self.decompressed_image[i, j] = np.clip(self.decompressed_image[i, j], 0, 255)

        print("Decompression completed!")

        # Calculate quality metrics
        mse = np.mean((self.original_image - self.decompressed_image) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255**2 / mse)

        print(f"\n{'='*60}")
        print(f"QUALITY METRICS")
        print(f"{'='*60}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")

    def save_binary_file(self, filename='compressed_image.bin'):
        """Save compressed data as binary file"""
        with open(filename, 'w') as f:
            f.write(self.binary_data)
        print(f"\nBinary file saved: {filename}")
        print(f"Binary file size: {len(self.binary_data)/8:.2f} bytes")

    def display_results(self):
        """Display all 6 images"""
        print(f"\n{'='*60}")
        print(f"DISPLAYING RESULTS")
        print(f"{'='*60}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Image Compression/Decompression Results', fontsize=16, fontweight='bold')

        # Original
        axes[0, 0].imshow(self.original_image, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        # Predicted
        axes[0, 1].imshow(self.predicted_image, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('2. Predicted Image')
        axes[0, 1].axis('off')

        # Error (normalized for visualization)
        axes[0, 2].imshow(self.error_image, cmap='gray')
        axes[0, 2].set_title('3. Error Image')
        axes[0, 2].axis('off')

        # Quantized (normalized for visualization)
        axes[1, 0].imshow(self.quantized_image, cmap='gray')
        axes[1, 0].set_title('4. Quantized Image')
        axes[1, 0].axis('off')

        # De-quantized (normalized for visualization)
        axes[1, 1].imshow(self.dequantized_image, cmap='gray')
        axes[1, 1].set_title('5. De-quantized Image')
        axes[1, 1].axis('off')

        # Decompressed
        axes[1, 2].imshow(self.decompressed_image, cmap='gray', vmin=0, vmax=255)
        axes[1, 2].set_title('6. Decompressed Image')
        axes[1, 2].axis('off')

        plt.tight_layout()

        # Save the figure
        plt.savefig('compression_results.png', dpi=150, bbox_inches='tight')
        print("Results saved as 'compression_results.png'")

        plt.show()

    def save_individual_images(self):
        """Save each image separately"""
        print("\nSaving individual images...")

        # Create output directory
        os.makedirs('output_images', exist_ok=True)

        # Save each image
        Image.fromarray(self.original_image.astype(np.uint8)).save('output_images/1_original.png')
        Image.fromarray(self.predicted_image.astype(np.uint8)).save('output_images/2_predicted.png')

        # Normalize error for saving
        error_normalized = ((self.error_image - self.error_image.min()) /
                           (self.error_image.max() - self.error_image.min()) * 255).astype(np.uint8)
        Image.fromarray(error_normalized).save('output_images/3_error.png')

        # Normalize quantized for saving
        quant_normalized = ((self.quantized_image - self.quantized_image.min()) /
                           (self.quantized_image.max() - self.quantized_image.min()) * 255).astype(np.uint8)
        Image.fromarray(quant_normalized).save('output_images/4_quantized.png')

        # Normalize dequantized for saving
        dequant_normalized = ((self.dequantized_image - self.dequantized_image.min()) /
                             (self.dequantized_image.max() - self.dequantized_image.min()) * 255).astype(np.uint8)
        Image.fromarray(dequant_normalized).save('output_images/5_dequantized.png')

        Image.fromarray(self.decompressed_image.astype(np.uint8)).save('output_images/6_decompressed.png')

        print("All images saved in 'output_images' folder")


def display_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print(" "*15 + "IMAGE COMPRESSION SYSTEM")
    print("="*60)
    print("\n1. Compression")
    print("2. Decompression")
    print("3. Exit")
    print("\n" + "="*60)


def main():
    """Main function"""
    compressor = ImageCompressionSystem()

    while True:
        display_menu()
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            # Compression
            image_path = input("\nEnter image path: ").strip()

            if not os.path.exists(image_path):
                print(f"Error: Image file '{image_path}' not found!")
                continue

            num_bits_input = input("Enter number of bits (default=2): ").strip()
            num_bits = int(num_bits_input) if num_bits_input else 2

            if num_bits < 1 or num_bits > 8:
                print("Error: Number of bits must be between 1 and 8!")
                continue

            # Compress
            compressor.compress_image(image_path, num_bits)

            # Decompress
            compressor.decompress_image()

            # Display results
            compressor.display_results()

            # Save individual images
            save_individual = input("\nSave individual images? (y/n): ").strip().lower()
            if save_individual == 'y':
                compressor.save_individual_images()

        elif choice == '2':
            # Decompression
            if compressor.original_image is None:
                print("\nError: No compressed image available!")
                print("Please compress an image first (Option 1)")
                continue

            compressor.decompress_image()
            compressor.display_results()

        elif choice == '3':
            # Exit
            print("\n" + "="*60)
            print("Thank you for using Image Compression System!")
            print("="*60 + "\n")
            break

        else:
            print("\nInvalid choice! Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
