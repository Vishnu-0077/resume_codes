# Motion Detection Using GPU

This project provides a highly optimized, GPU-accelerated pipeline for video processing, including motion detection, noise reduction, edge enhancement, and brightness adjustment. The core logic leverages NVIDIA CUDA-capable GPUs via CuPy and PyCUDA for significant speedup over CPU-based processing.

## How the Code Works

### 1. GPU Initialization
- The script first checks for CUDA-capable GPUs using PyCUDA.
- Device information is printed for user verification.

### 2. Video Processing Pipeline
The main function, `process_video_with_npp`, performs the following steps for each frame:

#### a. Frame Acquisition
- Reads frames from the input video using OpenCV.

#### b. Brightness Adjustment
- Each frame is converted to float32 and uploaded to the GPU.
- Brightness is adjusted by multiplying pixel values by a configurable factor.

#### c. Noise Reduction (Optional)
- If enabled, a fast Gaussian blur is applied using separable convolution.
- The Gaussian kernel is generated and applied on the GPU for efficiency.

#### d. Edge Enhancement (Optional)
- Sobel edge detection is performed on the GPU.
- The resulting edge map is added to the original frame to enhance edges.

#### e. Motion Detection (Optional)
- The difference between the current and previous frames is computed on the GPU.
- The difference is thresholded and morphologically processed to reduce noise.
- Contours are detected, and bounding boxes are drawn around moving regions.

#### f. Resizing (Optional)
- Frames can be resized before being written to the output video.

#### g. Output
- Processed frames are written to a new video file.
- Progress and estimated time remaining are displayed during processing.

### 3. Command-Line Interface
- The script can be run directly and accepts an optional input video path as a command-line argument.
- If no argument is provided, a default video path is used.

## How to Run

### 1. Install Dependencies

You need:
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- PyCUDA
- CuPy (version matching your CUDA toolkit, e.g., `cupy-cuda11x`)

Install with pip (replace `cupy-cuda11x` with your CUDA version):

```bash
pip install numpy opencv-python pycuda cupy-cuda11x
```

### 2. Run the Script

```bash
python motion_detection_using_gpu_main.py [input_video_path]
```

- If `input_video_path` is not provided, the script uses a default video file path (edit the script to change this default).
- The processed video will be saved as `processed_video_objdet.mp4` in the current directory.

### 3. Parameters

You can modify the following parameters in the `main()` function or by editing the script:
- `brightness_factor`: Adjusts frame brightness.
- `resize_scale`: Resizes frames (1.0 = no resize).
- `apply_noise_reduction`: Enables/disables noise reduction.
- `noise_reduction_strength`: Controls blur strength.
- `apply_edge_enhancement`: Enables/disables edge enhancement.
- `edge_enhancement_strength`: Controls edge enhancement strength.
- `apply_motion_detection`: Enables/disables motion detection.
- `motion_threshold`: Sensitivity for motion detection.
- `min_area_threshold`: Minimum area for detected motion regions.

## Notes

- Requires an NVIDIA GPU with CUDA support.
- For best performance, ensure your CUDA drivers and toolkit are up to date.
- If you encounter issues with CuPy installation, refer to the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
