import cv2
import numpy as np

# 1. Load images in grayscale
ref_img = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)
align_img = cv2.imread('align_this.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Define the motion model (e.g., Affine)
motion_model = cv2.MOTION_AFFINE

# 3. Allocate space for the warp matrix based on the motion model
if motion_model == cv2.MOTION_HOMOGRAPHY:
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else:
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# 4. Set termination criteria (Max 100 iterations or 1e-7 epsilon threshold)
iterations = 100
termination_eps = 1e-7
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, termination_eps)

# 5. Run the ECC algorithm wrapped in a try-except block
try:
    cc, warp_matrix = cv2.findTransformECC(ref_img, align_img, warp_matrix, motion_model, criteria)
    
    # 6. Apply the warp matrix to align the original image
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for 3x3 Homography matrices
        aligned_result = cv2.warpPerspective(align_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for 2x3 Translation, Euclidean, and Affine matrices
        aligned_result = cv2.warpAffine(align_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
    print(f"Alignment successful. Correlation Coefficient: {cc}")

except cv2.error as e:
    if e.code == cv2.Error.StsNoConv:
        print("Alignment failed: The algorithm did not converge within the designated iterations.")
    else:
        print(f"OpenCV Error occurred: {e}")
