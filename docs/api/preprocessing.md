# preprocessing video files


## Filtering of frames with broken buffers (blocks of a frame) 

### Broken buffer detection methods:

- **Local Contrast Analysis**

    - frame is divided into horizontal blocks based on `config.buffer_size` to ensure local contrast

    - second derivative in the **horizontal direction (x-axis)** is computed using `np.diff(block, n=2, axis=1)`, highlighting abrupt changes in pixel intensity across columns
        - more effective for spotting localized discontinuities compard to first derivative

    - second derivative in the **vertical direction (y-axis)** is computed using `np.diff(block, n=2, axis=0)`, detecting sudden changes in pixel intensity across rows

    - absolute mean of **both** second derivatives is calculated to quantify the level of noise in the block

    - If the mean contrast exceeds the noise detection threshold (`config.threshold`), the block is marked as noisy

    - binary mask is generated where noisy blocks are assigned a value of `1`, indicating regions with broken buffers

Example: If `config.threshold` is set to `5.0`, any block with a mean second 
        derivative value greater than `5.0` is considered a broken buffer.

### Types of buffer (blocks of a frame) errors 
for a 200x200 pixel frame

#### 1. Check-pattern
##### less than one row broken (<200 px)
![Image 1](../images/preprocess_broken_buffers/one_row1.png)
![Image 2](../images/preprocess_broken_buffers/one_row2.png)


##### several rows broken (2-19 rows of pixels)
![Image 3](../images/preprocess_broken_buffers/several_rows1.png)
![Image 4](../images/preprocess_broken_buffers/several_rows2.png)


##### one block broken (>19 rows of pixels)
![Image 5](../images/preprocess_broken_buffers/one_block1.png)
![Image 6](../images/preprocess_broken_buffers/one_block2.png)


##### several blocks broken
![Image 7](../images/preprocess_broken_buffers/several_blocks1.png)
![Image 8](../images/preprocess_broken_buffers/several_blocks2.png)



#### 2. Black-out pattern
##### less than one row broken (<200 px)
![Image 9](../images/preprocess_broken_buffers/black_one_block1.png)
![Image 10](../images/preprocess_broken_buffers/black_one_block2.png)


##### less than one row broken (<200 px)
![Image 11](../images/preprocess_broken_buffers/black_majority_frame1.png)

