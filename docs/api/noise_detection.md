# denoising / pre-processing

## Filter broken buffers (block of images) in a frame 

explain how to filter, different methods we have now
how to use them
testing preo-processing algorithms on 200x200 pixel frames

### Types of broken buffer (blocks of images)
#### 'Check pattern' broken buffers:
Black and white pixels

- **Less than one row broken ('check pattern' <200 px)**:


![Image 1](../images/preprocess_broken_buffers/onerow1.png)
![Image 2](../images/preprocess_broken_buffers/onerow2.png)

 
- **More than one row but clearly less than a block broken ('check pattern' 2-19 rows of pixels)**:


![Image 3](../images/preprocess_broken_buffers/2_19_rows1.png)
![Image 4](../images/preprocess_broken_buffers/2_19_rows2.png)

- **One block broken ('check-pattern' >19 rows of pixels)**:

![Image 5](../images/preprocess_broken_buffers/oneblock1.png)
![Image 6](../images/preprocess_broken_buffers/oneblock2.png)


- **Several blocks brocken ('check pattern' distinctive blocks)**:

![Image 7](../images/preprocess_broken_buffers/severalblocks1.png)
![Image 8](../images/preprocess_broken_buffers/severalblocks2.png)


#### Black-out broken buffers:
Entirely black pixels

- **One block broken ('black-out')**:

![Image 8](../images/preprocess_broken_buffers/black_oneblock1.png)
![Image 9](../images/preprocess_broken_buffers/black_oneblock2.png)

- **Majority of frame broken ('black-out')**:

![Image 9](../images/preprocess_broken_buffers/major_black_frame1.png)

##
add parts for spaitial mask filtering of horizontal stripes
