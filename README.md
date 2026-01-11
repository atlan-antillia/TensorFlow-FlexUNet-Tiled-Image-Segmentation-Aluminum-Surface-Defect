<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aluminum-Surface-Defect (2026/01/11)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Aluminum-Surface-Defect</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1V1dPmTL1CVyD66MFmYB_SzitJ1yM4y4O/view?usp=sharing">
<b>Tiled-Aluminum-Surface-Defect-ImageMask-Dataset.zip</b></a> which was derived by us from <br><br>
<b>aluminum dataset</b> in
<a href="https://figshare.com/articles/thesis/Codes_images_and_labels_of_the_paper_in_FDTransUnet_An_Aluminum_Surface_Defect_Segmentation_Model_Based_on_Feature_Differentiation_/27778473">
<b>Codes, images and labels of the paper in "FDTransUnet： An Aluminum Surface Defect Segmentation Model Based on Feature Differentiation"
</b></a> on figshare.com
<br><br>
<b>Divide-and-Conquer Strategy</b><br>
In this experiment with the TensorFlowFlexUNet segmentation model, 
since the images and masks in <b>aluminum dataset</b>  are very large (2560x1920 pixels),
we adopted the following <b>Divide-and-Conquer Strategy</b> for building the segmentation model.
<br><br>
<b>1. Tiled Image and Colorized Mask Dataset</b><br>
We generated a Resized image and colorized mask dataset of 2560x 2048 pixels from the original <b>aluminum dataset</b>
, and then generated a 512x512 pixels tiledly-split dataset from  the Resized one.
<!--
 by using <a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a>.<br>
 -->
<br><br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model by using the Tiled-Aluminum-Surface-Defect-ImageMask-Dataset.
<br>
<br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict defect regioins 
 for the mini_test images with a resolution of 2560x2048  pixels.<br><br>

<hr>
<b>Actual Image Segmentation for Aluminum-Surface-Defect Images of 2560x2048 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks
, but lack precision in some areas.
<br><br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/103.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/103.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/103.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/205.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/205.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/205.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/132.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/132.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/132.png" width="320" height="auto"></td>
</tr>
 
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<b>aluminum dataset</b> in
<a href="https://figshare.com/articles/thesis/Codes_images_and_labels_of_the_paper_in_FDTransUnet_An_Aluminum_Surface_Defect_Segmentation_Model_Based_on_Feature_Differentiation_/27778473">
<b>Codes, images and labels of the paper in "FDTransUnet： An Aluminum Surface Defect Segmentation Model Based on Feature Differentiation"
</b> </a> on figshare.com
<br><br>
The dataset contains 360  images and their corresponding labels of 2560x1920 pixels, and  the following six categories of defects:
<br>
<b>Missing paint</b>, <b>Surface bulge</b>, <b>Surface impurity</b>, <b>Pit</b>, <b>Wrinkle</b>, and <b>Coating crack</b>.
<br><br>
Please see also  
<a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320060">
FDTransUnet: An aluminum surface defect segmentation model based on feature differentiation
</a>
<br><br>
<b>Citation</b><br>
tang, mingzhu (2024). Codes, images and labels of the paper in <br>
"FDTransUnet： An Aluminum Surface Defect Segmentation Model Based on Feature Differentiation". figshare. <br>
Thesis. https://doi.org/10.6084/m9.figshare.27778473.v1
<br>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>
<br>
<br>
<h3>
2 Aluminum-Surface-Defect ImageMask Dataset
</h3>
 If you would like to train this Aluminum-Surface-Defect Segmentation model by yourself,
please down load our dataset <a href="https://drive.google.com/file/d/1V1dPmTL1CVyD66MFmYB_SzitJ1yM4y4O/view?usp=sharing">
<b>Tiled-Aluminum-Surface-Defect-ImageMask-Dataset.zip</b>
</a> on the google drive,
expand the downloaded , and put it under <b>./dataset/</b> to be.
<pre>
./dataset
└─Aluminum-Surface-Defect
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Aluminum-Surface-Defect Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/Aluminum-Surface-Defect_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br><br>
We also used the following color-class-mapping table to generate our colorized masks and define a rgb_map for our mask format
between indexed colors and rgb_colors.<br>
<br>
<a id="color-class-mapping-table"><b>Defects color class mapping table</b></a>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/Defect1.png' widith='40' height='25'></td><td>(255, 0, 0)</td><td>Defect1</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/Defect2.png' widith='40' height='25'></td><td>(0, 255, 0)</td><td>Defect2</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/Defect3.png' widith='40' height='25'></td><td>(0, 0, 255)</td><td>Defect3</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/Defect4.png' widith='40' height='25'></td><td>(255, 255, 0)</td><td>Defect4</td></tr>
<tr><td>5</td><td with='80' height='auto'><img src='./color_class_mapping/Defect5.png' widith='40' height='25'></td><td>(0, 255, 255)</td><td>Defect5</td></tr>
<tr><td>6</td><td with='80' height='auto'><img src='./color_class_mapping/Defect6.png' widith='40' height='25'></td><td>(255, 0, 255)</td><td>Defect6</td></tr>
</table>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Aluminum-Surface-Defect TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 7

base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Aluminum-Surface-Defect 8 classes.<br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<pre>
[mask]
mask_file_format = ".png"
;Aluminum-Surface-Defect 1+6
rgb_map {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3, (255, 255, 0): 4, (0, 255, 255): 5, (255, 0, 255): 6}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>
By using this epoch_change_tiled_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (25,26,27)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (50,51,52)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 52 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/train_console_output_at_epoch52.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Aluminum-Surface-Defect.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/evaluate_console_output_at_epoch52.png" width="880" height="auto">
<br><br>Image-Segmentation-Aluminum-Surface-Defect

<a href="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Aluminum-Surface-Defect/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.07
dice_coef_multiclass,0.967
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Aluminum-Surface-Defect.<br>
<pre>
>./4.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Aluminum-Surface-Defect  Images of 2560x2048 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks
, but lack precision in some areas.
<br>
<br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/324.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/324.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/324.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/182.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/182.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/296.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/296.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/296.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/161.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/161.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/161.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/26.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/26.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/26.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/images/247.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test/masks/247.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Aluminum-Surface-Defect/mini_test_output_tiled/247.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>
References
</h3>
<b>1.  Surface Defect Detection of Aluminum Profiles Based on Multiscale and Self-Attention Mechanisms</b><br>
Yichuan Shao, Shuo Fan, Qian Zhao, Le Zhang and Haijing Sun<br>
<a href="https://www.mdpi.com/1424-8220/24/9/2914">https://www.mdpi.com/1424-8220/24/9/2914</a>
<br><br>
<b>2. FDTransUnet: An aluminum surface defect segmentation model based on feature differentiation</b><br>
Mingzhu Tang ,Wencheng Wang<br>
<a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320060">https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320060</a>
<br>
<br>
<b>3. Surface Defect Detection for Aerospace Aluminum Profiles with Attention Mechanism and Multi-Scale Features</b><br>
 Yin-An Feng  and Wei-Wei Song<br>
<a href="https://www.mdpi.com/2079-9292/13/14/2861">https://www.mdpi.com/2079-9292/13/14/2861</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
