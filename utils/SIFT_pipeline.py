import numpy as np
from pathlib import Path
import cv2

class SIFTMatcher:
    """
    Feature extraction and matching using SIFT + RANSAC.
    
    Traditional computer vision pipeline:
    1. Extract SIFT features from all images
    2. Match features using ratio test (Lowe's)
    3. Geometric filtering with RANSAC
    """
    
    def __init__(self, images_dir: str, nfeatures: int = 10000):
        """
        Initialize SIFT matcher.
        
        Args:
            images_dir: Directory containing input images
            nfeatures: Maximum number of SIFT features per image
        """
        self.images_dir = Path(images_dir)
        self.nfeatures = nfeatures
        
        # Storage
        self.image_paths = []
        self.sift_features = {}
        self.sift = None
        
    def load_images(self):
        """Load all images from directory"""
        print("\nLoading images for SIFT...")
        
        # Reset and load fresh
        self.image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_paths = []
        for ext in extensions:
            all_paths.extend(self.images_dir.glob(ext))
        
        # Dedupe by absolute path (Windows is case-insensitive)
        dedup = {}
        for p in all_paths:
            dedup[str(p.resolve()).lower()] = p
        
        # Sort for reproducibility
        self.image_paths = sorted(dedup.values(), key=lambda p: str(p).lower())
        
        print(f"✓ Found {len(self.image_paths)} unique images")
        return len(self.image_paths)
    
    def extract_features(self):
        """Extract SIFT features from all loaded images"""
        print("\nExtracting SIFT features...")
        print("="*60)
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        
        # Extract features from all images
        self.sift_features = {}
        
        for img_path in self.image_paths:
            # Load image
            img = cv2.imread(str(img_path))
            
            # Detect and compute SIFT features
            keypoints, descriptors = self.sift.detectAndCompute(img, None)
            
            self.sift_features[img_path] = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image': img
            }
            
            print(f"  {img_path.name}: {len(keypoints)} keypoints")
        
        print("="*60)
        print(f"✓ SIFT features extracted from {len(self.sift_features)} images")
        
        return self.sift_features
    
    def find_matches_with_sift(self, img1_path: Path, img2_path: Path, 
                                ratio_threshold: float = 0.75,
                                ransac_reproj_threshold: float = 3.0):
        """
        Find matches between two images using SIFT + RANSAC.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            ratio_threshold: Lowe's ratio test threshold
            ransac_reproj_threshold: RANSAC reprojection threshold
            
        Returns:
            matches_im0, matches_im1: Nx2 arrays of matching points
            match_objects: List of cv2.DMatch objects
            fundamental_matrix: Estimated fundamental matrix
        """
        # Get features
        kp1 = self.sift_features[img1_path]['keypoints']
        desc1 = self.sift_features[img1_path]['descriptors']
        kp2 = self.sift_features[img2_path]['keypoints']
        desc2 = self.sift_features[img2_path]['descriptors']
        
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            # Not enough features
            return np.array([]), np.array([]), [], None
        
        # BFMatcher with ratio test
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            # Not enough matches for RANSAC
            return np.array([]), np.array([]), good_matches, None
        
        # Extract point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Find fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                          ransacReprojThreshold=ransac_reproj_threshold,
                                          confidence=0.99)
        
        if mask is None:
            # RANSAC failed
            return np.array([]), np.array([]), good_matches, None
        
        # Filter matches using inlier mask
        inlier_matches = [good_matches[idx] for idx, m in enumerate(mask) if m[0] == 1]
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]
        
        return inlier_pts1, inlier_pts2, inlier_matches, F
