# Image environment recognition tool



## Description

This repository is a collection of landscape recognition tools. There are:

- horizon detection (version 0.0.1)
- sun location (version 0.0.1)
- daytime detection (version 0.0.1)
- clouds detection (currently in implementation)
- whether/flow detection



## Installation and usage

Python 2.7 considered to be installed before start.

To install the package use [pip](http://www.pip-installer.org/).

```
$ pip install image-recognition-tool
```

Or download the repository. 

It is recommended to use virtual environment but it's not necessary.

To install dependencies

```
$ pip install -r requirements.txt
```

For usage run 

```
$ python -i /path/to/file/file_name.png
```

Either `-i` or  `--image` comman line arguments allowed.

**Note:** only *.png images are alowed supported.

For the development purpose or output information (as sun spot position and horizon lines) files `sun.py` and `horizon.py` may be used separately.



## Horizon Detection

Horizon detection is a problem of determining a line-divider between ground and sky on images. 

There are 3 commons ways to solve this problem:

- Machine learning (ML)
- Segmentation by light conditions
- Edge and line detection

Using ML could be highly accurate in a case of big size of a learning data set and quitly impossible for small tesing amonunt of particular images.

This program uses a combination of the 2 last approaches: segmentation and edge detection on a base of OpenCV2.

### Idea

The core assumption of light segmentation is to define sky as a "ligh" segment and ground as a "dark" segment. To segmentize an image uses [Otsu Binarization](https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html). This is an adaptive thresholding method which uses as arbitary number some approximately value between 2 peaks on histogram of bimodal image.

![](/https://drive.google.com/open?id=1om_0ySTqw1JapcLblRb26sGm6-dKTC9d)

After some [Morphological processing](https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html) with Erosion and Dilation methods typically the image has 2 solid determined lighting zones. Eventually, a part of the ligh segment contour is a horizon line.

![](/Users/Helen/Downloads/horizon_doc_3.png)



A bottleneck of this approach is unknowing information about processing image. Most algorithms adopted for good visability, contrast and sharpness level. Also Outsu binarization is based on assumption that each of lighting zone takes 50% of image, which is obviously not always the truth.

Edge detection methods uses as a second tier to improve algorithm accurancy . The idea of this method to determine edges on the image according to contrast changes. [Canny Edge detection](https://docs.opencv.org/3.4.1/da/d22/tutorial_py_canny.html) is one of the most known method in this area. The method output is binary image of found edges.

![](/Users/Helen/Downloads/horizon_doc_5.png)

Then [Hough Line transform](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html) uses to determine lines on the edge image.

![](/Users/Helen/Downloads/horizon_doc_6.png)



The final step is to choose the most relevant lines as horizon assumption and add all missing lines to complete whole horizon line.

![Filtered lines](/Users/Helen/Downloads/horizon_doc_7.png) 



### Implementation details

#### Image processing

A lot of image recognition problems are highly time sensative. Tasks could requre some immediate decision which looses it's relevancy very fast. This is reasonable for a lot of robotics problem related to robot self directing and real-time decision making. 

Otherwise image processing tasks usually have high time complexity because of numerous by-pixiel operations. This is why image **resize** is the first obvious algorithm improvement for scalable solutions. 

`resized = cv2.resize(img,(0, 0), None, .25, .25)`

Edge detecting algorithms are highly noise-sensative. Resizing helps to reduce unwanted edges and to determine distinctive edges better. Here the example of Canny for the same image but different sizes:

`canny = cv2.Canny(img,30,150)`

![](/Users/Helen/Downloads/horizon_doc_9.png)

Another important transformation step is smoothing the image. This step is highly valuable before edge detecting procedures. The main reason is noise reducing, however, not the only one.  Commonly to use linear filters, in which processig pixel recalculates as weighted sum of some input pixels values (mask or kernel)

![g(i,j) = \sum_{k,l} f(i+k, j+l) h(k,l)](https://docs.opencv.org/2.4.13.4/_images/math/63c07a05324e75528d0c1cf509df4519ae9f7e9e.png)

Although, the most useful filter is [Gaussian](https://docs.opencv.org/2.4.13.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html) which is also used here. It's based on the normal distribution and "reweight" every pixel in accordance to it.

![G_{0}(x, y) = A  e^{ \dfrac{ -(x - \mu_{x})^{2} }{ 2\sigma^{2}_{x} } +  \dfrac{ -(y - \mu_{y})^{2} }{ 2\sigma^{2}_{y} } }](https://docs.opencv.org/2.4.13.4/_images/math/1fafabfdfb9d1057c61a51d9dcf97ca958262aeb.png)



The only addition parameter is required in OpenCV Gaussian filters it is a processing kernal size.

`blur = cv2.GaussianBlur(img,(5,5),0)`

Most provided in OpenCV algorithms works only for 1-channel images. So RGB to GRAY transformation is a typical step of image processing. However, that also improves time results, decreasing amount of further processing data.

RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B 

`gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`

Sometimes input images could have different light or visibility condition and some additional adjustment is required to standartize an image for the algorithms. OpenCV library has a wide range of tools to improve image main indexes, such as brightness or contrast. Usuallly these methods are based on [histogram equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html). Among them are Histograms Equalization and CLAHE (Contrast Limited Adaptive Histogram Equalization) 

`equ = cv2.equalizeHist(img)`

![](/Users/Helen/Downloads/horizon_doc_10.png)

`#create a CLAHE object (Arguments are optional).`

`clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`

`cl1 = clahe.apply(img)`

![](/Users/Helen/Downloads/horizon_doc_11.png)

Unfornutunately, it's not always working for good in image processing, so this step is optional and depends on a type of images.

There are some cases when noise and pixel anomalities is not an issue. Moreover, sometimes increasing of sharpness level can help to improve method results. 

Histogram equlization change sharpness in a smart way, but it's possible also to do it manually, using [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)).

`kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])`

 `gray = cv2.filter2D(gray, -1, kernel)`

The idea in emphasizing edge pixels, adding more weight.

#### Thresholding

Thresholding is the basic segmentation method. An OTSU method is based on histogram processing to find minimum between foreground and background. Binary thresholding methods usually requires additional parameters as a `threshold` value and `maxValue`.

`(param, thresh) = cv2.threshold(gray, threshold, maxValue, cv2.THRESH_BINARY+cv2.THRESH_OTSU)`

Otsu method returns `threshold` value for input image and output binary segmentized image.

Usually that binary images are very noisy. That's why morphological transformation is important step of posprocessing for better segmentation. 

There are 2 types of "smoothing" binary images: reduce white and reduce black. The most common to use them in pair: once white noise on black is reduced typically useful to reduce also a black noise on white.  OpenCV provides Erosion-Dilation and Opening-Closing pairs. Both of pairs use kernal approach, assuming random oposite color pixels as a noise. Process can also repeats in several iterations. 

The pair Erode-Dilate uses more "agressive" strategy and extends color segment.

![](/Users/Helen/Downloads/horizon_doc_12.png)

Otherwise Opening-Closing just reduces the noise.

![](/Users/Helen/Downloads/horizon_doc_13.png)

Basically, Erode and Dilate methods are more useful in case of segmentation.

`erode = cv2.erode(binary_img, kernel, iterations=4)`
`dilate = cv2.dilate(erode, kernel, iterations=3)` 

![](/Users/Helen/Downloads/horizon_doc_14.png)

But it's still not always useful for horizon problem. There some cases where sky part is darker or ground part is not dark enough or if the ground part is missing at all.

![](/Users/Helen/Downloads/horizon_doc_15.png)



#### Contour extracting

The next step is extracting light segment contour and keep its bottom part. OpenCV provides a [function](https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html) for this.

`im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)`

There are three arguments in **cv2.findContours()** function, first one is source image, second is contour retrieval mode, third is contour approximation method. And it outputs the contours and hierarchy. contours is a Python list of all the contours in the image.

It is also possible to calculate segment [moment](https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html) and filter points along segment side.

The other way to extract contour is manual. There are some props and cons of this methods. Usually, all library methods are highly effective and it's highly uneffective to reach pixels data directly. Especially this is the truth for Python OpenCV implementation, because of numpy realization. 

But it's quite useful for this problem. Because of uneven segment contour, it is much harder to extract appropriate points from **cv2.findContours()** output. This is why the enforced way to operate with image from top to bottom and extract edge points from the top of dark segment. And then filter them as one contour line. Which is still quite uneffective.

![](/Users/Helen/Downloads/horizon_doc_16.png)

#### Edge and line detection

Other alternative approach for horizon line determining is an edge detection. This can be implemented as a combination of 2 methods: Canny Edge Detector + Hough Line  Transform. 

[Canny edge detector](https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html) build an edge map of 1-channel image. The output depends on the threshold arguments and filtering more weak results.

`cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) `

There are some common ways to calculates proper threshold1, threshold2 image values. For example, using Otsu threshhold parameter.

`high_thresh, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`

`lowThresh = 0.5*high_thresh`

Or directly calculate mean and standart image array deviation.

`(mu, sigma) = cv2.meanStdDev(gray)`

`canny = cv2.Canny(gray, mu - sigma, mu + sigma)`

This works great for the cases where all the edges need to be detected. 

But sometimes this approach filters much more than needed for low contrast images, thus it might be useful in general case to run Canny with lower bottom threshold, despite of noisy results.

`(mu, sigma) =  (array([[139.75723298]]), array([[85.26942198]]))`

`(mu - sigma, mu + sigma) = (54.5, 225)` 

`canny = cv2.Canny(gray,30,150)`

![](/Users/Helen/Downloads/horizon_doc_18.png)



As long as OpenCV Canny method returns map image, edge results are needed to be further detected in more useful view. [Hough Line](https://en.wikipedia.org/wiki/Hough_transform) Transform works great for this case. 

OpenCV has 2 related functions : standart **HoughLines()** and probabilistic [**HoughLinesP()**](https://docs.opencv.org/3.4.0/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb). The last one is being used here. 

`lines=cv.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])`

In general, standart function HoughLines() is more accurate and faster. It is useful for cases with obvious strict lines (buildings, road lines, horizon directions, angle detection), because of the ouput as an array of two-element vectors (ρ,θ) . ρ is the distance from the coordinate origin (0,0) (top-left corner of the image).

HoughLinesP() returns unsorted array of line segments represented as 2 points. Which is quite useful for uneven line building. The method works much better for good parameters adjustment.

In best cases Hough transform returns easy to process result, but more often it doesn't.

![](/Users/Helen/Downloads/horizon_doc_20.png)



#### Hybrid approach and line filtering

Final horizon detection step is to filter Hough Transform output along the threshold method result and connect all the lines together in some contour.

There could be a lot of different approaches to filter lines. Especially, when any kind of data about the picture angle or objects inside (trees, mountains, buildings etc.) is unknown in advanced. 

In this repository used a simple filtering based on threshold contour and angles and it is not always accurate, so this is open question for improvements.

![Edge detection](/Users/Helen/Downloads/horizon_doc_21.png)

![Thresholding](/Users/Helen/Downloads/horizon_doc_22.png)

![](/Users/Helen/Downloads/horizon_doc_23.png)



Looking at this example, it might be unclear why even use editional method if threshold works better. This is probably the truth for a lot of contrasted pictures with good visability. But it's not that useful for unevenly divided or with no ground part at all and many other pictures. That's why it's always better to have some additional approach just to make sure it works.

![](/Users/Helen/Downloads/horizon_doc_24.png) 

## Sun Location

Sun location is a problem of recognition sun location on picture. There are several interpretations of the problem meaning, and solving approach depends on this interpretation.  For example, it's unclear if algorithm should detect a visible half of sun or partially hidden sun. 

The common known solutions are:

- Machine Learching technics (requires big amount of data)
- Combination of bright zone detection and edge detection
- Combination of bright zone detection and circle detection

The last one is being used here. This approach assumes the sun is bright round object in a light picture segment.

### Implementation detail

The are a couple step:

- detect most bright zone on a picture
- make sure it belongs to sun

To detect the most brightest zone uses Threshold methods

`thresh = cv2.threshold(gray, 250, 255,  cv2.THRESH_BINARY)[1]`

The assumption here to define sun in (250, 255) bright zone, which is optional and probably can wary. This helps to exclude from further steps pictures with worse lighting condidions.

![](/Users/Helen/Downloads/sun_doc_1.png)

And vise versa, to emphasize those pictures where sun position is more obvious.

![](/Users/Helen/Downloads/sun_doc_2.png)

As long as sun sun location essentially above ground, next step is thresholding dark zone with lower threshold value (200) to leave some more space for circle detection method.

`thresh_2 = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO)[1]`

![](/Users/Helen/Downloads/sun_doc_3.png)



For circle detection [Hough Circle](https://en.wikipedia.org/wiki/Circle_Hough_Transform) Transform is being used. It's based on a similar as Hough Line Transform idea. There are 2 stages of the algorithms: 1) find potential center 2) find radius. OpenCV library provides the implementation of this method and requires additional [parameters.](https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/)

` cv2.HoughCircles(thresh_2,cv2.HOUGH_GRADIENT,1,20,param1,param2,minRadius,maxRadius)`

Unfortunately, radius search usually less accurate, but center detection works much better. 

Despite on noise filtering, a lot of unwanted circles are being detected anyway.

![](/Users/Helen/Downloads/sun_doc_4.png)

And more often small circles left undetected.

![](/Users/Helen/Downloads/sun_doc_8.png)

One of the possible solution filtering is to check if most of (250,255) thresholded area is inside of observing circle. Because if consider circle is somewhere around the sun it suppose to include the whole sun zone. 

![](/Users/Helen/Downloads/sun_doc_6.png)

Otherwise, this approach will filter those pictures where sun is somewhere inside big bright zone

![](/Users/Helen/Downloads/sun_doc_7.png)



## Daytime check

In general daytime recognition is a deep problem of detecting possible time when the picture is taken.

This could be accurately realized as ML program using big amount of pictures for any possible time/light/landscape.

Besides, in this repository used the simplest possible mehod for time check. It based on assumption of good lghtining conditions. Daytime recognized in accordance to the chosen by Otsu threshold algorithm parameter. Moreover, the parameters are not accurate and chosen experimentally. 



## License

This repository uses the [MIT License](https://github.com/exercism/python/blob/master/LICENSE).



## Resources

1. A quick algorithm for horizon line detection in marine images. Tomasz Praczyk (https://link.springer.com/article/10.1007/s00773-017-0464-8)
2. http://vision.stanford.edu/teaching/cs131_fall1314_nope/lectures/lecture5_edges_cs131.pdf
3. https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
4. https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
5. Vision-based Horizon Detection and Target Tracking for UAVs Yingju Chen, Ahmad Abushakra and Jeongkyu Lee Department of Computer Science and Engineering, University of Bridgeport (https://pdfs.semanticscholar.org/3219/c3c6b38e4a36f70e2a13f27adb957ba76de1.pdf)
6. An Efficient Sky Detection Algorithm Based on Hybrid Probability Model Chi-Wei Wang, Jian-Jiun Ding *, and Po-Jen Chen Graduate Institute of Communication Engineering, National Taiwan University (http://www.apsipa.org/proceedings_2015/pdf/254.pdf)
7. New Approach to Thresholding and Contour Detection for Object Surface Inspection in Machine Vision K. Židek, A. Hošovský, J. Dubják (https://pdfs.semanticscholar.org/2c5b/020676b9950f68109054dc14fa3ad0669038.pdf)
8. Rain or Snow Detection in Image Sequences through use of a Histogram of Orientation of Streaks Jérémie Bossu ·Nicolas Hautière ·Jean-Philippe Tarel (https://www.researchgate.net/publication/220660048_Rain_or_Snow_Detection_in_Image_Sequences_Through_Use_of_a_Histogram_of_Orientation_of_Streaks)

 



