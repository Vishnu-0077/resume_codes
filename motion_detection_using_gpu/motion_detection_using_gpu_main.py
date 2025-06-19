import os
import sys
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
import time
import cupy as cp

def check_gpu_availability():
    """Check if CUDA GPU is available and print device information."""
    try:
        device_count = cuda.Device.count()
        if device_count == 0:
            print("No CUDA-capable GPU found.")
            return False
        
        print(f"Found {device_count} CUDA-capable GPU(s):")
        for i in range(device_count):
            device = cuda.Device(i)
            print(f"  GPU {i}: {device.name()}")
            print(f"    Compute Capability: {device.compute_capability()[0]}.{device.compute_capability()[1]}")
            print(f"    Total Memory: {device.total_memory() // (1024**2)} MB")
        return True
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False

def process_video_with_npp(input_video_path, output_video_path, 
                          brightness_factor=1.2, 
                          resize_scale=1.0,
                          apply_noise_reduction=True,
                          apply_edge_enhancement=True,
                          noise_reduction_strength=1.0,
                          edge_enhancement_strength=0.3,
                          apply_motion_detection=False,
                          motion_threshold=0.03,
                          min_area_threshold=100):
    """
    Process video with GPU acceleration
    - Optimized for speed with vectorized operations
    - Added motion detection with bounding boxes
    """
    try:
        # Check if input file exists
        if not os.path.isfile(input_video_path):
            print(f"Input video file not found: {input_video_path}")
            return False
            
        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video_path}")
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} at {fps} FPS, {frame_count} frames")
        
        # Calculate new dimensions if resizing
        if resize_scale != 1.0:
            new_width = int(width * resize_scale)
            new_height = int(height * resize_scale)
            output_dimensions = (new_width, new_height)
            print(f"Resizing to: {new_width}x{new_height}")
        else:
            output_dimensions = (width, height)
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, output_dimensions)
        
        # Import cupy for GPU operations
        try:
            import cupy as cp
        except ImportError:
            print("Error: CuPy is required for GPU operations. Install with 'pip install cupy-cuda11x' (replace 11x with your CUDA version).")
            return False
        
        # Define CUDA kernel for fast Gaussian blur (using separable kernel)
        if apply_noise_reduction:
            # Create 1D Gaussian kernel (separable for efficiency)
            sigma = noise_reduction_strength
            kernel_size = max(3, int(2 * round(2 * sigma) + 1))  ##calculate kernel size
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            # Create horizontal and vertical kernels on GPU
            k_half = kernel_size // 2
            x = cp.arange(-k_half, k_half + 1, dtype=cp.float32) #creates a 1d array of indices with x or kernel size
            # Gaussian function
            gaussian_kernel_1d = cp.exp(-0.5 * (x / sigma) ** 2)
            gaussian_kernel_1d = gaussian_kernel_1d / gaussian_kernel_1d.sum()
            
            # For separable convolution
            gaussian_kernel_h = gaussian_kernel_1d.reshape(1, -1)
            gaussian_kernel_v = gaussian_kernel_1d.reshape(-1, 1)
            
        # Define Sobel kernels for edge detection
        if apply_edge_enhancement:
            sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
            sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
        
        # Function to apply Gaussian blur using separable convolution
        def fast_separable_blur(image):
            """Apply Gaussian blur using separable convolution (much faster)"""
            # Handle multi-channel images
            if len(image.shape) == 3:
                result = cp.zeros_like(image)
                # Process each channel
                for c in range(image.shape[2]):
                    # Horizontal pass
                    temp = cv2.filter2D(cp.asnumpy(image[:,:,c]), -1, cp.asnumpy(gaussian_kernel_h))# we are extracting the color channels and converting them from cupy to numpy
                    # Vertical pass on result of horizontal
                    result[:,:,c] = cp.asarray(cv2.filter2D(temp, -1, cp.asnumpy(gaussian_kernel_v))) #cv.filter produces applies the 2d convolutional layer(slides the kernel ) and give a new pixel value
                return result
            else:
                # Horizontal pass
                temp = cv2.filter2D(cp.asnumpy(image), -1, cp.asnumpy(gaussian_kernel_h))
                # Vertical pass on result of horizontal
                return cp.asarray(cv2.filter2D(temp, -1, cp.asnumpy(gaussian_kernel_v)))
        
        # Function to apply edge detection using OpenCV and GPU for speed
        def fast_edge_detection(image):
            """Apply edge detection using OpenCV and GPU for speed"""
            # Handle multi-channel images
            if len(image.shape) == 3:
                edge_image = cp.zeros_like(image)
                # Process each channel
                for c in range(image.shape[2]):
                    # Use OpenCV for the Sobel operations (faster)
                    img_np = cp.asnumpy(image[:,:,c])
                    grad_x = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3) #applies sobel kernel and specifies output data type as 32-bit float
                    grad_y = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
                    
                    # Calculate gradient magnitude
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                    
                    # Normalize gradient magnitude,(Normalizes the gradient magnitude to the range [0, 1]. This ensures that the edge strength is represented consistently.)
                    if np.max(grad_mag) > 0:
                        grad_mag = grad_mag / np.max(grad_mag)
                    
                    edge_image[:,:,c] = cp.asarray(grad_mag)
                return edge_image
            else:
                # Use OpenCV for the Sobel operations (faster)
                img_np = cp.asnumpy(image)
                grad_x = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
                
                # Calculate gradient magnitude
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                # Normalize gradient magnitude
                if np.max(grad_mag) > 0:
                    grad_mag = grad_mag / np.max(grad_mag)
                
                return cp.asarray(grad_mag)
        
        # Function to detect motion between frames
        def detect_motion(current_frame, prev_frame, threshold=0.03, min_area=100):
            """Detect motion between frames and return bounding boxes.
            
            Args:
                current_frame: CuPy array of current frame (float32, 0-1 range)
                prev_frame: CuPy array of previous frame (float32, 0-1 range)
                threshold: Minimum difference to consider as motion
                min_area: Minimum contour area to consider a valid motion region
                
            Returns:
                list of bounding boxes (x, y, w, h) for regions with motion
            """
            if prev_frame is None:
                return []
                
            # Convert to grayscale if color
            if len(current_frame.shape) == 3:
                current_gray = cp.dot(current_frame[..., :3], cp.array([0.299, 0.587, 0.114], dtype=cp.float32))
                prev_gray = cp.dot(prev_frame[..., :3], cp.array([0.299, 0.587, 0.114], dtype=cp.float32))
            else:
                current_gray = current_frame
                prev_gray = prev_frame
                
            # Calculate absolute difference
            diff = cp.abs(current_gray - prev_gray)
            
            # Convert to NumPy array and scale to uint8 range
            diff_np = cp.asnumpy(diff) # Convert to NumPy for OpenCV. we are converting to numpy bcoz we are unable to scale using cupy
            diff_np_scaled = (diff_np * 255).astype(np.uint8)  # Scale and convert to uint8
            
            # Apply a small blur to reduce noise
            diff_blurred = cv2.GaussianBlur(diff_np_scaled, (5, 5), 0)
            
            # Threshold the difference (ensure threshold is appropriate for uint8 scale)
            _, thresh = cv2.threshold(diff_blurred, int(threshold * 255), 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up noise and fill holes
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and create bounding boxes
            bounding_boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))
                    
            return bounding_boxes
        
        processed_frames = 0
        start_time = time.time()
        last_update_time = start_time
        last_update_frame = 0
        
        # Initialize variables for motion detection
        prev_frame_gpu = None
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            
            # Convert frame to float32 for processing and upload to GPU in one step
            frame_gpu = cp.asarray(frame.astype(np.float32) / 255.0)
            
            # Store a copy of the frame before processing for motion detection
            if apply_motion_detection:
                current_frame_for_motion = cp.copy(frame_gpu)
            
            # 1. Apply brightness adjustment (vectorized)
            frame_gpu = cp.clip(frame_gpu * brightness_factor, 0, 1)
            
            # 2. Apply noise reduction if enabled
            if apply_noise_reduction:
                # Use OpenCV's optimized implementation via our helper function
                frame_gpu = fast_separable_blur(frame_gpu)
            
            # 3. Apply edge enhancement if enabled
            if apply_edge_enhancement:
                edge_image = fast_edge_detection(frame_gpu)
                frame_gpu = cp.clip(frame_gpu + edge_enhancement_strength * edge_image, 0, 1)
            
            # 4. Motion detection if enabled
            motion_boxes = []
            if apply_motion_detection and prev_frame_gpu is not None:
                motion_boxes = detect_motion(
                    current_frame_for_motion,
                    prev_frame_gpu,
                    threshold=motion_threshold,
                    min_area=min_area_threshold
                )
            
            # Update previous frame for next iteration
            if apply_motion_detection:
                prev_frame_gpu = current_frame_for_motion
            
            # 5. Resize the frame if needed
            if resize_scale != 1.0:
                # Convert to numpy array for OpenCV resize
                temp_cpu = cp.asnumpy(frame_gpu)
                resized_cpu = cv2.resize(temp_cpu, output_dimensions)
                # Only transfer back to GPU if more processing is needed
                if processed_frames == frame_count:
                    processed_frame = resized_cpu
                else:
                    frame_gpu = cp.asarray(resized_cpu)
                    processed_frame = cp.asnumpy(frame_gpu)
                    
                # Scale motion boxes if resize is applied
                if resize_scale != 1.0 and motion_boxes:
                    scaled_boxes = [] 
                    for x, y, w, h in motion_boxes:
                        scaled_boxes.append((
                            int(x * resize_scale),
                            int(y * resize_scale),
                            int(w * resize_scale),
                            int(h * resize_scale)
                        ))
                    motion_boxes = scaled_boxes
            else:
                # Download result back to CPU
                processed_frame = cp.asnumpy(frame_gpu)
            
            # Convert back to uint8 for OpenCV
            processed_frame = (processed_frame * 255).astype(np.uint8)
            
            # Draw motion bounding boxes
            for x, y, w, h in motion_boxes:
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Update progress at most once per second (reduces console spam)
            current_time = time.time()
            if current_time - last_update_time >= 1.0:
                elapsed = current_time - start_time
                frames_since_last = processed_frames - last_update_frame
                recent_fps = frames_since_last / (current_time - last_update_time)
                
                # Calculate estimated time remaining
                if recent_fps > 0:
                    frames_remaining = frame_count - processed_frames
                    eta_seconds = frames_remaining / recent_fps
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    
                    print(f"Processed {processed_frames}/{frame_count} frames ({recent_fps:.2f} FPS) - ETA: {eta_min}m {eta_sec}s")
                else:
                    print(f"Processed {processed_frames}/{frame_count} frames")
                    
                last_update_time = current_time
                last_update_frame = processed_frames
                
        # Release resources
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time
        print(f"Video processing complete. Processed {processed_frames} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
        print(f"Output saved to {output_video_path}")
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Check for command line arguments
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
    else:
        input_video_path = "/home/vishnu/documents/anitum/stock-footage-people-working-at-business-office-and-sitting-by-tables-with-computers-businesswoman-chats-by.webm"
    
    output_video_path = "processed_video_objdet.mp4"
    
    # Check GPU availability
    if not check_gpu_availability():
        print("Error: GPU not available for processing.")
        return
    
    # Process video with all enhancements
    success = process_video_with_npp(
        input_video_path, 
        output_video_path,
        brightness_factor=1.2,         # Increase brightness by 20%
        resize_scale=1.0,              # No resize
        apply_noise_reduction=True,    # Apply noise reduction
        noise_reduction_strength=0.1,  # Standard noise reduction (higher = more blur)
        apply_edge_enhancement=True,   # Apply edge enhancement
        edge_enhancement_strength=0.6, # Edge enhancement strength (higher = stronger edges)
        apply_motion_detection=True,   # Apply motion detection
        motion_threshold=0.03,        # Motion detection threshold
        min_area_threshold=400        # Minimum contour area for motion detection
    )
    
    if success:
        print(f"Video successfully processed and saved to {output_video_path}")
    else:
        print("Video processing failed.")

if __name__ == "__main__":
    main()
