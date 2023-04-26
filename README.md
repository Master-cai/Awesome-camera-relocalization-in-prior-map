# Awesome-camera-relocalization-in-prior-map

A list of visual(camera) re-localization researches. Visual relocalization refers to the problem of estimating the position and orientation using the information of an existing prior map based on the image captured from visual sensors. We sort out  these methods according to the type of map, mainly including image databases and point cloud maps.



[toc]

## Surveys



## Visual relocalization in Image Database Maps

### Image Retrieval

#### Image retrieval methods

![](https://img.shields.io/badge/year-2001-g)|![](https://img.shields.io/badge/pub-IJCV-orange)|[Modeling the shape of the scene: A holistic representation of the spatial envelope](https://link.springer.com/article/10.1023/A:1011139631724)

![](https://img.shields.io/badge/year-2008-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[IM2GPS: estimating geographic information from a single image](https://ieeexplore.ieee.org/abstract/document/4587784/)

![](https://img.shields.io/badge/year-2010-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[Aggregating local descriptors into a compact image representation](https://ieeexplore.ieee.org/abstract/document/5540039/) |[code](https://github.com/raulmur/ORB_SLAM2)

![](https://img.shields.io/badge/year-2011-g)|![](https://img.shields.io/badge/pub-IROS-orange)|[Real-time loop detection with bags of binary words](https://ieeexplore.ieee.org/abstract/document/6094885/) 

![](https://img.shields.io/badge/year-2011-g)|![](https://img.shields.io/badge/pub-ICCVW-orange)|[Automatic alignment of paintings and photographs depicting a 3d scene](https://ieeexplore.ieee.org/abstract/document/6130291/) 

![](https://img.shields.io/badge/year-2014-g)|![](https://img.shields.io/badge/pub-ICRA-orange)|[Fast relocalisation and loop closing in keyframe-based slam](https://ieeexplore.ieee.org/abstract/document/6906953/) 

![](https://img.shields.io/badge/year-2015-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[24/7 place recognition by view synthesis](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Torii_247_Place_Recognition_2015_CVPR_paper.html) 

![](https://img.shields.io/badge/year-2016-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[NetVLAD: cnn architecture for weakly supervised place recognition](http://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html) |[code](https://github.com/Relja/netvlad)![](https://img.shields.io/github/stars/Relja/netvlad?style=social)

![](https://img.shields.io/badge/year-2016-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[CNN image retrieval learns from bow: unsupervised fine-tuning with hard examples](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_1) |[code](https://github.com/filipradenovic/cnnimageretrieval-pytorch)

![](https://img.shields.io/badge/year-2016-g)|![](https://img.shields.io/badge/pub-BMVC-orange)|[Filtering 3d keypoints using gist for accurate image-based localization](http://www.bmva.org/bmvc/2016/papers/paper127/paper127.pdf) 

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-TRO-orange)|[Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras](https://ieeexplore.ieee.org/abstract/document/7946260/) |[code](https://github.com/raulmur/ORB_SLAM2)

![](https://img.shields.io/badge/year-2018-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[RelocNet: continuous metric learning relocalisation using neural nets](http://openaccess.thecvf.com/content_ECCV_2018/papers/Vassileios_Balntas_RelocNet_Continous_Metric_ECCV_2018_paper.pdf) 

![](https://img.shields.io/badge/year-2019-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[NeXtVLAD: an efficient neural network to aggregate frame-level features for large-scale video classification](https://openaccess.thecvf.com/content_eccv_2018_workshops/w22/html/Lin_NeXtVLAD_An_Efficient_Neural_Network_to_Aggregate_Frame-level_Features_for_ECCVW_2018_paper.html) |[code](https://github.com/linrongc/youtube-8m)

![](https://img.shields.io/badge/year-2019-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[NeXtVLAD: an efficient neural network to aggregate frame-level features for large-scale video classification](https://openaccess.thecvf.com/content_eccv_2018_workshops/w22/html/Lin_NeXtVLAD_An_Efficient_Neural_Network_to_Aggregate_Frame-level_Features_for_ECCVW_2018_paper.html) |[code](https://github.com/linrongc/youtube-8m)![](https://img.shields.io/github/stars/linrongc/youtube-8m?style=social)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[TransGeo: transformer is all you need for cross-view image geo-localization](http://openaccess.thecvf.com/content/CVPR2022/html/Zhu_TransGeo_Transformer_Is_All_You_Need_for_Cross-View_Image_Geo-Localization_CVPR_2022_paper.html) |[code](https://github.com/jeff-zilence/transgeo2022)![](https://img.shields.io/github/stars/jeff-zilence/transgeo2022?style=social)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[Deep visual geo-localization benchmark](http://openaccess.thecvf.com/content/CVPR2022/html/Berton_Deep_Visual_Geo-Localization_Benchmark_CVPR_2022_paper.html) |[code](https://github.com/gmberton/deep-visual-geo-localization-benchmark)![](https://img.shields.io/github/stars/gmberton/deep-visual-geo-localization-benchmark?style=social)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-RAL-orange)|[An efficient and scalable collection of fly-inspired voting units for visual place recognition in changing environments](https://ieeexplore.ieee.org/abstract/document/9672749/)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-RAL-orange)|[An efficient and scalable collection of fly-inspired voting units for visual place recognition in changing environments](https://ieeexplore.ieee.org/abstract/document/9672749/)

#### Feature matching and pose estimation

![](https://img.shields.io/badge/year-2006-g)|![](https://img.shields.io/badge/pub-3DPVT-orange)|[Image based localization in urban environments](https://ieeexplore.ieee.org/abstract/document/4155707/)

![](https://img.shields.io/badge/year-2014-g)|![](https://img.shields.io/badge/pub-TPAMI-orange)|[Image geo-localization based on multiplenearest neighbor feature matching usinggeneralized graphs](https://ieeexplore.ieee.org/abstract/document/6710175/)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[Are large-scale 3d models really necessary for accurate visual localization](http://openaccess.thecvf.com/content_cvpr_2017/html/Sattler_Are_Large-Scale_3D_CVPR_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-ICCVW-orange)|[Camera relocalization by computing pairwise relative poses using convolutional neural network](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Laskar_Camera_Relocalization_by_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2020-g)|![](https://img.shields.io/badge/pub-Arxiv-orange)|[S2dnet: Learning accurate correspondences for sparse-to-dense feature matching](https://arxiv.org/abs/2004.01673) |[code](https://github.com/germain-hug/S2DNet-Minimal)![](https://img.shields.io/github/stars/germain-hug/S2DNet-Minimal?style=social)

![](https://img.shields.io/badge/year-2020-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[SuperGlue: learning feature matching with graph neural networks](http://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) |[code](https://github.com/magicleap/SuperGluePretrainedNetwork)![](https://img.shields.io/github/stars/magicleap/SuperGluePretrainedNetwork?style=social)

![](https://img.shields.io/badge/year-2020-g)|![](https://img.shields.io/badge/pub-ICRA-orange)|[To learn or not to learn: visual localization from essential matrices](https://ieeexplore.ieee.org/abstract/document/9196607/)

![](https://img.shields.io/badge/year-2021-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[LoFTR: detector-free local feature matching with transformers](http://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html) |[code](https://github.com/zju3dv/LoFTR)![](https://img.shields.io/github/stars/zju3dv/LoFTR?style=social)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[Clustergnn: Cluster-based coarse-to-fine graph neural network for efficient feature matching](http://openaccess.thecvf.com/content/CVPR2022/html/Shi_ClusterGNN_Cluster-Based_Coarse-To-Fine_Graph_Neural_Network_for_Efficient_Feature_Matching_CVPR_2022_paper.html)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[3DG-stfm: 3d geometric guided student-teacher feature matching](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_8) |[code](https://github.com/ryan-prime/3dg-stfm)![](https://img.shields.io/github/stars/ryan-prime/3dg-stfm?style=social)

#### Image appearance normalization

![](https://img.shields.io/badge/year-2019-g)|![](https://img.shields.io/badge/pub-ICRA-orange)|[Night-to-day image translation for retrieval-based localization](https://ieeexplore.ieee.org/abstract/document/8794387/) |[code](https://github.com/AAnoosheh/ToDayGAN)![](https://img.shields.io/github/stars/AAnoosheh/ToDayGAN?style=social)

![](https://img.shields.io/badge/year-2020-g)|![](https://img.shields.io/badge/pub-ICRA-orange)|[Adversarial feature disentanglement for place recognition across changing appearance](https://ieeexplore.ieee.org/abstract/document/9196518/)

![](https://img.shields.io/badge/year-2021-g)|![](https://img.shields.io/badge/pub-KBS-orange)|[A deep learning based image enhancement approach for autonomous driving at night](https://www.sciencedirect.com/science/article/pii/S0950705120307462)

### Pose Regression

#### monocular camera

![](https://img.shields.io/badge/year-2015-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](http://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html) |[code](https://github.com/alexgkendall/caffe-posenet)![](https://img.shields.io/github/stars/alexgkendall/caffe-posenet?style=social)

![](https://img.shields.io/badge/year-2016-g)|![](https://img.shields.io/badge/pub-ICRA-orange)|[Modelling uncertainty in deep learning for camera relocalization](https://ieeexplore.ieee.org/abstract/document/7487679) |[code](https://github.com/alexgkendall/caffe-posenet)![](https://img.shields.io/github/stars/alexgkendall/caffe-posenet?style=social)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[Geometric Loss Functions for Camera Pose Regression With Deep Learning](https://openaccess.thecvf.com/content_cvpr_2017/html/Kendall_Geometric_Loss_Functions_CVPR_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[Image-based localization using hourglass networks](http://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Melekhov_Image-Based_Localization_Using_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[ Image-based localization using lstms for structured feature correlation](http://openaccess.thecvf.com/content_iccv_2017/html/Walch_Image-Based_Localization_Using_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-ICRA-orange)|[Delving deeper into convolutional neural networks for camera relocalization](https://ieeexplore.ieee.org/abstract/document/7989663)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-IROS-orange)|[Deep regression for monocular camera-based 6-dof global localization in outdoor environments](https://ieeexplore.ieee.org/abstract/document/8205957/)

![](https://img.shields.io/badge/year-2019-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[Prior guided dropout for robust visual localization in dynamic environments](http://openaccess.thecvf.com/content_ICCV_2019/html/Huang_Prior_Guided_Dropout_for_Robust_Visual_Localization_in_Dynamic_Environments_ICCV_2019_paper.html) |[code](https://github.com/zju3dv/RVL-Dynamic)![](https://img.shields.io/github/stars/zju3dv/RVL-Dynamic?style=social)

![](https://img.shields.io/badge/year-2020-g)|![](https://img.shields.io/badge/pub-AAAI-orange)|[Atloc: Attention guided camera localization](https://ojs.aaai.org/index.php/AAAI/article/view/6608) |[code](https://github.com/BingCS/AtLoc)![](https://img.shields.io/github/stars/BingCS/AtLoc?style=social)

![](https://img.shields.io/badge/year-2020-g)|![](https://img.shields.io/badge/pub-CVPRW-orange)|[Extending absolute pose regression to multiple scenes](http://openaccess.thecvf.com/content_CVPRW_2020/html/w3/Blanton_Extending_Absolute_Pose_Regression_to_Multiple_Scenes_CVPRW_2020_paper.html) |[code](https://github.com/yolish/multi-scene-pose-transformer)![](https://img.shields.io/github/stars/yolish/multi-scene-pose-transformer?style=social)

![](https://img.shields.io/badge/year-2021-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[Learning multi-scene absolute pose regression with transformers](http://openaccess.thecvf.com/content/ICCV2021/html/Shavit_Learning_Multi-Scene_Absolute_Pose_Regression_With_Transformers_ICCV_2021_paper.html) |[code](https://github.com/yolish/multi-scene-pose-transformer)![](https://img.shields.io/github/stars/yolish/multi-scene-pose-transformer?style=social)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[DiffPoseNet: direct differentiable camera pose estimation](http://openaccess.thecvf.com/content/CVPR2022/html/Parameshwara_DiffPoseNet_Direct_Differentiable_Camera_Pose_Estimation_CVPR_2022_paper.html)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[Dfnet: Enhance absolute pose regression with direct feature matching](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_1) |[code](https://github.com/activevisionlab/dfnet)![](https://img.shields.io/github/stars/activevisionlab/dfnet?style=social)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[Sc-wls: Towards interpretable feed-forward camera re-localization](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_34) |[code](https://github.com/xinwu98/sc-wls)![](https://img.shields.io/github/stars/xinwu98/sc-wls?style=social)

#### sequence images

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-ACIVS-orange)|[Relative camera pose estimation using convolutional neural networks](https://link.springer.com/chapter/10.1007/978-3-319-70353-4_57) |[code](https://github.com/AaltoVision/relativeCameraPose)![](https://img.shields.io/github/stars/AaltoVision/relativeCameraPose?style=social)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[VidLoc: a deep spatio-temporal model for 6-dof video-clip relocalization](http://openaccess.thecvf.com/content_cvpr_2017/html/Clark_VidLoc_A_Deep_CVPR_2017_paper.html)

![](https://img.shields.io/badge/year-2018-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[Geometry-aware learning of maps for camera localization](http://openaccess.thecvf.com/content_cvpr_2018/html/Brahmbhatt_Geometry-Aware_Learning_of_CVPR_2018_paper.html) |[code](https://github.com/NVlabs/geomapnet)![](https://img.shields.io/github/stars/NVlabs/geomapnet?style=social)

![](https://img.shields.io/badge/year-2018-g)|![](https://img.shields.io/badge/pub-ICRA-orange)|[Deep auxiliary learning for visual localization and odometry](https://ieeexplore.ieee.org/abstract/document/8462979/) |[code](https://github.com/decayale/vlocnet)![](https://img.shields.io/github/stars/decayale/vlocnet?style=social)

![](https://img.shields.io/badge/year-2018-g)|![](https://img.shields.io/badge/pub-RAL-orange)|[VLocNet++: deep multitask learning for semantic visual localization and odometry](https://ieeexplore.ieee.org/abstract/document/8458420/)

![](https://img.shields.io/badge/year-2019-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[Local supports global: deep camera relocalization with sequence enhancement](http://openaccess.thecvf.com/content_ICCV_2019/html/Xue_Local_Supports_Global_Deep_Camera_Relocalization_With_Sequence_Enhancement_ICCV_2019_paper.html)

![](https://img.shields.io/badge/year-2022-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[GTCaR: Graph Transformer for Camera Re-localization](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_14)

#### RGBD image

![](https://img.shields.io/badge/year-2018-g)|![](https://img.shields.io/badge/pub-TASE-orange)|[Indoor relocalization in challenging environments with dual-stream convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/7869254)

## Visual relocalization in Point Cloud Maps

### Feature based method(Visual Point Cloud)

#### F2P(Feature to Point) and P2F(Point to Feature)

![](https://img.shields.io/badge/year-2007-g)|![](https://img.shields.io/badge/pub-IJCV-orange)|[Monocular vision for mobile robot localization and autonomous navigation](https://hal.science/hal-01635679/)

![](https://img.shields.io/badge/year-2009-g)|![](https://img.shields.io/badge/pub-CVPR-orange)|[From structure-from-motion point clouds to fast location recognition](https://ieeexplore.ieee.org/abstract/document/5206587/)

![](https://img.shields.io/badge/year-2010-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[Location recognition using prioritized feature matching](https://link.springer.com/chapter/10.1007/978-3-642-15552-9_57)

![](https://img.shields.io/badge/year-2011-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[Fast image-based localization using direct 2d-to-3d matching](https://ieeexplore.ieee.org/abstract/document/6126302/)

![](https://img.shields.io/badge/year-2012-g)|![](https://img.shields.io/badge/pub-ECCV-orange)|[Worldwide pose estimation using 3d point clouds](https://link.springer.com/chapter/10.1007/978-3-319-25781-5_8)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-ICCV-orange)|[Efficient global 2d-3d matching for camera localization in a large-scale 3d map](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Efficient_Global_2D-3D_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)|![](https://img.shields.io/badge/pub-TPAMI-orange)|[Efficient & effective prioritized matching for large-scale image-based localization](https://ieeexplore.ieee.org/abstract/document/7572201/)

#### Improved matching method

