"""
Image Segmentation Module for Transparent Background Generation

This module applies trained segmentation models to create PNG images with transparent backgrounds.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Optional, List
import sys
import cv2

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ImageSegmenter:
    def create_transparent_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Create RGBA image with transparent background (mask=0 is transparent).
        Args:
            image: Original RGB image
            mask: Binary segmentation mask
        Returns:
            RGBA image with transparency
        """
        image = image.convert('RGB')
        arr = np.array(image)
        if mask.shape != arr.shape[:2]:
            from PIL import Image as PILImage
            mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((arr.shape[1], arr.shape[0]), Image.BILINEAR)
            mask = (np.array(mask_img) / 255.0 > 0.5).astype(np.uint8)
        # Zero out RGB where mask==0 to ensure background is transparent black
        arr = arr * mask[..., None]
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack([arr, alpha])
        return Image.fromarray(rgba, mode='RGBA')
    
    """
    Segments images using trained models and generates transparent backgrounds.
    """
    
    def __init__(self, model_path: str, model_type: str = "U-NET", device: Optional[str] = None):
        """
        Initialize the segmenter.
        
        Args:
            model_path: Path to trained segmentation model
            model_type: Type of model ("U-NET", "DeepLabV3Plus", "SegFormer", "SegNet", "MaskFormer")
            device: Device to use ("cuda" or "cpu"). Auto-detected if None.
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✓ Loaded {model_type} model from {model_path}")
    
    def _load_model(self):
        """Load the appropriate model architecture and weights"""
        
        if self.model_type == "U-NET":
            from utils.models.uNet import UNet
            channels = [32, 64, 128, 256, 512]
            model = UNet(in_channels=3, out_channels=1, channels=channels, 
                        bilinear=True, use_batchnorm=True)
        
        elif self.model_type == "DeepLabV3Plus":
            from utils.models.deeplabv3p import DeepLabV3Plus
            model = DeepLabV3Plus(num_classes=1, output_stride=16, backbone_width_mult=1.0)
        
        elif self.model_type == "SegFormer":
            from utils.models.SegFormer import segformer
            model = segformer(in_channels=3, num_classes=1)
        
        elif self.model_type == "SegNet":
            from utils.models.SegNet import segnet
            model = segnet(in_channels=3, num_classes=1, pretrained=False)
        
        elif self.model_type == "MaskFormer":
            from utils.models.maskFormer import MaskFormer
            from utils.models.resnet101 import resnet101_backbone
            
            resnet101 = resnet101_backbone()
            model = MaskFormer(
                backbone=resnet101,
                num_classes=1,
                num_queries=5,
                embed_dim=64,
                transformer_layers=1,
                transformer_heads=2,
                transformer_ffn_dim=256,
                return_binary=True
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Convert to tensor
        img_tensor = TF.to_tensor(image)
        
        # Normalize
        img_tensor = TF.normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict_mask(self, image: Image.Image, threshold: float = 0.5, use_contours: bool = True) -> np.ndarray:
        """
        Predict segmentation mask for an image using contour-based approach.
        
        Args:
            image: PIL Image in RGB format
            threshold: Threshold for binary segmentation (default: 0.5)
            use_contours: Use contour-based refinement (default: True)
            
        Returns:
            Binary mask as numpy array [H, W]
        """
        # Convert PIL to cv2 format
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_orig, w_orig = image_cv2.shape[:2]
        
        # Resize to 512x512 for model
        image_resized = cv2.resize(image_cv2, (512, 512), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = TF.to_tensor(image_rgb)
        image_tensor = TF.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_logits = self.model(image_tensor)
                pred_mask = (torch.sigmoid(pred_logits) > threshold).float().cpu().squeeze().numpy()
        
        if use_contours:
            # Find contours on mask (512x512)
            mask_uint8 = (pred_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Upscale contours to original image size
            scale_x = w_orig / 512.0
            scale_y = h_orig / 512.0
            upscaled_contours = [np.array([[int(pt[0][0]*scale_x), int(pt[0][1]*scale_y)] for pt in contour], dtype=np.int32) for contour in contours]
            
            # Create mask for original image
            mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)
            cv2.drawContours(mask_orig, upscaled_contours, -1, 255, thickness=cv2.FILLED)
            mask = (mask_orig / 255.0).astype(np.uint8)
        else:
            # Simple resize approach
            mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((w_orig, h_orig), Image.BILINEAR)
            mask = (np.array(mask_img) / 255.0 > threshold).astype(np.uint8)
        
        return mask
    
    
    def segment_image(self, input_path: str, output_path: str, threshold: float = 0.5, 
                     refine: bool = True, kernel_size: int = 5, return_mask: bool = False):
        """
        Segment a single image and save with transparent background.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output PNG
            threshold: Segmentation threshold (default: 0.5)
            refine: Apply morphological refinement (default: True)
            kernel_size: Size of morphological kernel for refinement (default: 5)
            return_mask: If True, also return the mask
            
        Returns:
            mask (optional): Binary segmentation mask if return_mask=True
        """
        # Load image
        image = Image.open(input_path).convert('RGB')
        
        # Predict mask using contour approach
        mask = self.predict_mask(image, threshold, use_contours=True)
        
        # Refine mask if requested
        if refine:
            mask = self.refine_mask(mask, kernel_size)
        
        # Create transparent-background composite image (RGBA)
        transparent_img = self.create_transparent_image(image, mask)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transparent_img.save(output_path, 'PNG')
        if return_mask:
            return mask
    
    def segment_directory(self, input_dir: str, output_dir: str, threshold: float = 0.5, 
                         refine: bool = True, kernel_size: int = 5, verbose: bool = True):
        """
        Segment all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output PNGs
            threshold: Segmentation threshold (default: 0.5)
            refine: Apply morphological refinement (default: True)
            kernel_size: Size of morphological kernel for refinement (default: 5)
            verbose: Print progress
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if verbose:
            print(f"Processing {len(image_files)} images with contour-based segmentation...")
        
        # Process each image
        for i, img_file in enumerate(image_files):
            output_path = output_dir / f"{img_file.stem}.png"
            
            try:
                self.segment_image(str(img_file), str(output_path), threshold, refine, kernel_size)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(image_files)} images")
            
            except Exception as e:
                print(f"✗ Error processing {img_file.name}: {e}")
                continue
        
        if verbose:
            print(f"✓ Completed segmentation of {len(image_files)} images")
            print(f"  Output directory: {output_dir}")
    
    def refine_mask(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Refine mask using morphological operations.
        
        Args:
            mask: Binary mask
            kernel_size: Size of morphological kernel
            
        Returns:
            Refined mask
        """
        import cv2
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def segment_with_refinement(self, input_path: str, output_path: str, 
                               threshold: float = 0.5, refine: bool = True, kernel_size: int = 5):
        """
        Segment image with optional mask refinement.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output PNG
            threshold: Segmentation threshold (default: 0.5)
            refine: Apply morphological refinement (default: True)
            kernel_size: Size of refinement kernel (default: 5)
        """
        # Load and predict using contour approach
        image = Image.open(input_path).convert('RGB')
        mask = self.predict_mask(image, threshold, use_contours=True)
        
        # Refine mask if requested
        if refine:
            mask = self.refine_mask(mask, kernel_size)

        # Create transparent-background image
        transparent_img = self.create_transparent_image(image, mask)
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transparent_img.save(output_path, 'PNG')


def segment_filtered_images(filtered_images_dir: str, 
                            model_path: str,
                            output_dir: str,
                            model_type: str = "U-NET",
                            threshold: float = 0.5,
                            refine: bool = True,
                            kernel_size: int = 5):
    """
    Convenience function to segment all filtered images.
    
    Args:
        filtered_images_dir: Directory with filtered images from preprocessing
        model_path: Path to trained segmentation model
        output_dir: Output directory for segmented images with transparency
        model_type: Type of segmentation model (default: U-NET)
        threshold: Segmentation threshold (default: 0.5)
        refine: Apply morphological refinement (default: True)
        kernel_size: Size of morphological kernel (default: 5)
    """
    print("="*60)
    print("Starting Image Segmentation with Contour-Based Approach")
    print("="*60)
    
    segmenter = ImageSegmenter(model_path, model_type)
    
    # Process all images with contour-based segmentation
    segmenter.segment_directory(
        input_dir=filtered_images_dir,
        output_dir=output_dir,
        threshold=threshold,
        refine=refine,
        kernel_size=kernel_size,
        verbose=True
    )
    
    print("="*60)
    print("✓ Segmentation completed!")
    print("="*60)


if __name__ == "__main__":
    # Example usage
    segment_filtered_images(
        filtered_images_dir="../reconstructions/InstantNGP_preprocessed/filtered_images",
        model_path="../models/U-NET_seg.pt",
        output_dir="../reconstructions/InstantNGP_preprocessed/segmented_images",
        model_type="U-NET"
    )
