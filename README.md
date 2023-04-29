# Awesome-camera-relocalization-in-prior-map

A list of visual(camera) re-localization researches. Visual relocalization refers to the problem of estimating the position and orientation using the information of an existing prior map based on the image captured from visual sensors. We sort out  these methods according to the type of map, mainly including image databases and point cloud maps.

- [Awesome-camera-relocalization-in-prior-map](#awesome-camera-relocalization-in-prior-map)
  - [Surveys](#surveys)
  - [Visual relocalization in Image Database Maps](#visual-relocalization-in-image-database-maps)
    - [Image Retrieval](#image-retrieval)
      - [Image retrieval methods](#image-retrieval-methods)
      - [Feature matching and pose estimation](#feature-matching-and-pose-estimation)
      - [Image appearance normalization](#image-appearance-normalization)
    - [Pose Regression](#pose-regression)
      - [monocular camera](#monocular-camera)
      - [sequence images](#sequence-images)
      - [RGBD image](#rgbd-image)
  - [Visual relocalization in Point Cloud Maps](#visual-relocalization-in-point-cloud-maps)
    - [Feature based method(Visual Point Cloud)](#feature-based-methodvisual-point-cloud)
      - [F2P(Feature to Point) and P2F(Point to Feature)](#f2pfeature-to-point-and-p2fpoint-to-feature)
      - [Improved matching method](#improved-matching-method)
      - [2D-3D pose estimation](#2d-3d-pose-estimation)
    - [3D $\\rightarrow$ 2D: projection methods](#3d-rightarrow-2d-projection-methods)
    - [2D $\\rightarrow$ 3D: Scene Dimensional Upgrading](#2d-rightarrow-3d-scene-dimensional-upgrading)
      - [Scene Coordinate Regression](#scene-coordinate-regression)
      - [point cloud Reconstruction](#point-cloud-reconstruction)
  - [Visual relocalization in Dense Boundary Representation Maps](#visual-relocalization-in-dense-boundary-representation-maps)
    - [Mesh map](#mesh-map)
    - [Surfel Map](#surfel-map)
    - [SDF Map](#sdf-map)
  - [Visual relocalization in high definition(HD) Map](#visual-relocalization-in-high-definitionhd-map)
  - [Visual relocalization in Semantic Maps](#visual-relocalization-in-semantic-maps)
    - [Semantic Global  Features](#semantic-global--features)
    - [Semantic Feature matching](#semantic-feature-matching)
    - [Robust Feature Selection](#robust-feature-selection)
  - [Visual relocalization in NeRF map](#visual-relocalization-in-nerf-map)
  - [Dataset](#dataset)
    - [Outdoor](#outdoor)
    - [Indoor](#indoor)
    - [Misc](#misc)


## Surveys



## Visual relocalization in Image Database Maps

### Image Retrieval

#### Image retrieval methods

![](https://img.shields.io/badge/year-2001-g)![](https://img.shields.io/badge/pub-IJCV-orange)[Modeling the shape of the scene: A holistic representation of the spatial envelope](https://link.springer.com/article/10.1023/A:1011139631724)

![](https://img.shields.io/badge/year-2008-g)![](https://img.shields.io/badge/pub-CVPR-orange)[IM2GPS: estimating geographic information from a single image](https://ieeexplore.ieee.org/abstract/document/4587784/)

![](https://img.shields.io/badge/year-2010-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Aggregating local descriptors into a compact image representation](https://ieeexplore.ieee.org/abstract/document/5540039/) |[code](https://github.com/raulmur/ORB_SLAM2)

![](https://img.shields.io/badge/year-2011-g)![](https://img.shields.io/badge/pub-IROS-orange)[Real-time loop detection with bags of binary words](https://ieeexplore.ieee.org/abstract/document/6094885/) 

![](https://img.shields.io/badge/year-2011-g)![](https://img.shields.io/badge/pub-ICCVW-orange)[Automatic alignment of paintings and photographs depicting a 3d scene](https://ieeexplore.ieee.org/abstract/document/6130291/) 

![](https://img.shields.io/badge/year-2014-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Fast relocalisation and loop closing in keyframe-based slam](https://ieeexplore.ieee.org/abstract/document/6906953/) 

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-CVPR-orange)[24/7 place recognition by view synthesis](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Torii_247_Place_Recognition_2015_CVPR_paper.html) 

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-CVPR-orange)[NetVLAD: cnn architecture for weakly supervised place recognition](http://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html) |[code](https://github.com/Relja/netvlad)![](https://img.shields.io/github/stars/Relja/netvlad?style=social)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-ECCV-orange)[CNN image retrieval learns from bow: unsupervised fine-tuning with hard examples](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_1) |[code](https://github.com/filipradenovic/cnnimageretrieval-pytorch)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-BMVC-orange)[Filtering 3d keypoints using gist for accurate image-based localization](http://www.bmva.org/bmvc/2016/papers/paper127/paper127.pdf) 

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-TRO-orange)[Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras](https://ieeexplore.ieee.org/abstract/document/7946260/) |[code](https://github.com/raulmur/ORB_SLAM2)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-ECCV-orange)[RelocNet: continuous metric learning relocalisation using neural nets](http://openaccess.thecvf.com/content_ECCV_2018/papers/Vassileios_Balntas_RelocNet_Continous_Metric_ECCV_2018_paper.pdf) 

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ECCV-orange)[NeXtVLAD: an efficient neural network to aggregate frame-level features for large-scale video classification](https://openaccess.thecvf.com/content_eccv_2018_workshops/w22/html/Lin_NeXtVLAD_An_Efficient_Neural_Network_to_Aggregate_Frame-level_Features_for_ECCVW_2018_paper.html) |[code](https://github.com/linrongc/youtube-8m)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ECCV-orange)[NeXtVLAD: an efficient neural network to aggregate frame-level features for large-scale video classification](https://openaccess.thecvf.com/content_eccv_2018_workshops/w22/html/Lin_NeXtVLAD_An_Efficient_Neural_Network_to_Aggregate_Frame-level_Features_for_ECCVW_2018_paper.html) |[code](https://github.com/linrongc/youtube-8m)![](https://img.shields.io/github/stars/linrongc/youtube-8m?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-CVPR-orange)[TransGeo: transformer is all you need for cross-view image geo-localization](http://openaccess.thecvf.com/content/CVPR2022/html/Zhu_TransGeo_Transformer_Is_All_You_Need_for_Cross-View_Image_Geo-Localization_CVPR_2022_paper.html) |[code](https://github.com/jeff-zilence/transgeo2022)![](https://img.shields.io/github/stars/jeff-zilence/transgeo2022?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Deep visual geo-localization benchmark](http://openaccess.thecvf.com/content/CVPR2022/html/Berton_Deep_Visual_Geo-Localization_Benchmark_CVPR_2022_paper.html) |[code](https://github.com/gmberton/deep-visual-geo-localization-benchmark)![](https://img.shields.io/github/stars/gmberton/deep-visual-geo-localization-benchmark?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-RAL-orange)[An efficient and scalable collection of fly-inspired voting units for visual place recognition in changing environments](https://ieeexplore.ieee.org/abstract/document/9672749/)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-RAL-orange)[An efficient and scalable collection of fly-inspired voting units for visual place recognition in changing environments](https://ieeexplore.ieee.org/abstract/document/9672749/)

#### Feature matching and pose estimation

![](https://img.shields.io/badge/year-2006-g)![](https://img.shields.io/badge/pub-3DPVT-orange)[Image based localization in urban environments](https://ieeexplore.ieee.org/abstract/document/4155707/)

![](https://img.shields.io/badge/year-2014-g)![](https://img.shields.io/badge/pub-TPAMI-orange)[Image geo-localization based on multiplenearest neighbor feature matching usinggeneralized graphs](https://ieeexplore.ieee.org/abstract/document/6710175/)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Are large-scale 3d models really necessary for accurate visual localization](http://openaccess.thecvf.com/content_cvpr_2017/html/Sattler_Are_Large-Scale_3D_CVPR_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICCVW-orange)[Camera relocalization by computing pairwise relative poses using convolutional neural network](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Laskar_Camera_Relocalization_by_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-Arxiv-orange)[S2dnet: Learning accurate correspondences for sparse-to-dense feature matching](https://arxiv.org/abs/2004.01673) |[code](https://github.com/germain-hug/S2DNet-Minimal)![](https://img.shields.io/github/stars/germain-hug/S2DNet-Minimal?style=social)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-CVPR-orange)[SuperGlue: learning feature matching with graph neural networks](http://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) |[code](https://github.com/magicleap/SuperGluePretrainedNetwork)![](https://img.shields.io/github/stars/magicleap/SuperGluePretrainedNetwork?style=social)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-ICRA-orange)[To learn or not to learn: visual localization from essential matrices](https://ieeexplore.ieee.org/abstract/document/9196607/)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-CVPR-orange)[LoFTR: detector-free local feature matching with transformers](http://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html) |[code](https://github.com/zju3dv/LoFTR)![](https://img.shields.io/github/stars/zju3dv/LoFTR?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Clustergnn: Cluster-based coarse-to-fine graph neural network for efficient feature matching](http://openaccess.thecvf.com/content/CVPR2022/html/Shi_ClusterGNN_Cluster-Based_Coarse-To-Fine_Graph_Neural_Network_for_Efficient_Feature_Matching_CVPR_2022_paper.html)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ECCV-orange)[3DG-stfm: 3d geometric guided student-teacher feature matching](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_8) |[code](https://github.com/ryan-prime/3dg-stfm)![](https://img.shields.io/github/stars/ryan-prime/3dg-stfm?style=social)

#### Image appearance normalization

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Night-to-day image translation for retrieval-based localization](https://ieeexplore.ieee.org/abstract/document/8794387/) |[code](https://github.com/AAnoosheh/ToDayGAN)![](https://img.shields.io/github/stars/AAnoosheh/ToDayGAN?style=social)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Adversarial feature disentanglement for place recognition across changing appearance](https://ieeexplore.ieee.org/abstract/document/9196518/)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-KBS-orange)[A deep learning based image enhancement approach for autonomous driving at night](https://www.sciencedirect.com/science/article/pii/S0950705120307462)

### Pose Regression

#### monocular camera

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-ICCV-orange)[PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](http://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html) |[code](https://github.com/alexgkendall/caffe-posenet)![](https://img.shields.io/github/stars/alexgkendall/caffe-posenet?style=social)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Modelling uncertainty in deep learning for camera relocalization](https://ieeexplore.ieee.org/abstract/document/7487679) |[code](https://github.com/alexgkendall/caffe-posenet)![](https://img.shields.io/github/stars/alexgkendall/caffe-posenet?style=social)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Geometric Loss Functions for Camera Pose Regression With Deep Learning](https://openaccess.thecvf.com/content_cvpr_2017/html/Kendall_Geometric_Loss_Functions_CVPR_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Image-based localization using hourglass networks](http://openaccess.thecvf.com/content_ICCV_2017_workshops/w17/html/Melekhov_Image-Based_Localization_Using_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICCV-orange)[ Image-based localization using lstms for structured feature correlation](http://openaccess.thecvf.com/content_iccv_2017/html/Walch_Image-Based_Localization_Using_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Delving deeper into convolutional neural networks for camera relocalization](https://ieeexplore.ieee.org/abstract/document/7989663)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-IROS-orange)[Deep regression for monocular camera-based 6-dof global localization in outdoor environments](https://ieeexplore.ieee.org/abstract/document/8205957/)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Prior guided dropout for robust visual localization in dynamic environments](http://openaccess.thecvf.com/content_ICCV_2019/html/Huang_Prior_Guided_Dropout_for_Robust_Visual_Localization_in_Dynamic_Environments_ICCV_2019_paper.html) |[code](https://github.com/zju3dv/RVL-Dynamic)![](https://img.shields.io/github/stars/zju3dv/RVL-Dynamic?style=social)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-AAAI-orange)[Atloc: Attention guided camera localization](https://ojs.aaai.org/index.php/AAAI/article/view/6608) |[code](https://github.com/BingCS/AtLoc)![](https://img.shields.io/github/stars/BingCS/AtLoc?style=social)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-CVPRW-orange)[Extending absolute pose regression to multiple scenes](http://openaccess.thecvf.com/content_CVPRW_2020/html/w3/Blanton_Extending_Absolute_Pose_Regression_to_Multiple_Scenes_CVPRW_2020_paper.html) |[code](https://github.com/yolish/multi-scene-pose-transformer)![](https://img.shields.io/github/stars/yolish/multi-scene-pose-transformer?style=social)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Learning multi-scene absolute pose regression with transformers](http://openaccess.thecvf.com/content/ICCV2021/html/Shavit_Learning_Multi-Scene_Absolute_Pose_Regression_With_Transformers_ICCV_2021_paper.html) |[code](https://github.com/yolish/multi-scene-pose-transformer)![](https://img.shields.io/github/stars/yolish/multi-scene-pose-transformer?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-CVPR-orange)[DiffPoseNet: direct differentiable camera pose estimation](http://openaccess.thecvf.com/content/CVPR2022/html/Parameshwara_DiffPoseNet_Direct_Differentiable_Camera_Pose_Estimation_CVPR_2022_paper.html)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Dfnet: Enhance absolute pose regression with direct feature matching](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_1) |[code](https://github.com/activevisionlab/dfnet)![](https://img.shields.io/github/stars/activevisionlab/dfnet?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Sc-wls: Towards interpretable feed-forward camera re-localization](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_34) |[code](https://github.com/xinwu98/sc-wls)![](https://img.shields.io/github/stars/xinwu98/sc-wls?style=social)

#### sequence images

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ACIVS-orange)[Relative camera pose estimation using convolutional neural networks](https://link.springer.com/chapter/10.1007/978-3-319-70353-4_57) |[code](https://github.com/AaltoVision/relativeCameraPose)![](https://img.shields.io/github/stars/AaltoVision/relativeCameraPose?style=social)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-CVPR-orange)[VidLoc: a deep spatio-temporal model for 6-dof video-clip relocalization](http://openaccess.thecvf.com/content_cvpr_2017/html/Clark_VidLoc_A_Deep_CVPR_2017_paper.html)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Geometry-aware learning of maps for camera localization](http://openaccess.thecvf.com/content_cvpr_2018/html/Brahmbhatt_Geometry-Aware_Learning_of_CVPR_2018_paper.html) |[code](https://github.com/NVlabs/geomapnet)![](https://img.shields.io/github/stars/NVlabs/geomapnet?style=social)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Deep auxiliary learning for visual localization and odometry](https://ieeexplore.ieee.org/abstract/document/8462979/) |[code](https://github.com/decayale/vlocnet)![](https://img.shields.io/github/stars/decayale/vlocnet?style=social)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-RAL-orange)[VLocNet++: deep multitask learning for semantic visual localization and odometry](https://ieeexplore.ieee.org/abstract/document/8458420/)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Local supports global: deep camera relocalization with sequence enhancement](http://openaccess.thecvf.com/content_ICCV_2019/html/Xue_Local_Supports_Global_Deep_Camera_Relocalization_With_Sequence_Enhancement_ICCV_2019_paper.html)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ECCV-orange)[GTCaR: Graph Transformer for Camera Re-localization](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_14)

#### RGBD image

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-TASE-orange)[Indoor relocalization in challenging environments with dual-stream convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/7869254)

## Visual relocalization in Point Cloud Maps

### Feature based method(Visual Point Cloud)

#### F2P(Feature to Point) and P2F(Point to Feature)

![](https://img.shields.io/badge/year-2007-g)![](https://img.shields.io/badge/pub-IJCV-orange)[Monocular vision for mobile robot localization and autonomous navigation](https://hal.science/hal-01635679/)

![](https://img.shields.io/badge/year-2009-g)![](https://img.shields.io/badge/pub-CVPR-orange)[From structure-from-motion point clouds to fast location recognition](https://ieeexplore.ieee.org/abstract/document/5206587/)

![](https://img.shields.io/badge/year-2010-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Location recognition using prioritized feature matching](https://link.springer.com/chapter/10.1007/978-3-642-15552-9_57)

![](https://img.shields.io/badge/year-2011-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Fast image-based localization using direct 2d-to-3d matching](https://ieeexplore.ieee.org/abstract/document/6126302/)

![](https://img.shields.io/badge/year-2012-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Worldwide pose estimation using 3d point clouds](https://link.springer.com/chapter/10.1007/978-3-319-25781-5_8)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Efficient global 2d-3d matching for camera localization in a large-scale 3d map](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Efficient_Global_2D-3D_ICCV_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-TPAMI-orange)[Efficient & effective prioritized matching for large-scale image-based localization](https://ieeexplore.ieee.org/abstract/document/7572201/)

#### Improved matching method

![](https://img.shields.io/badge/year-2006-g)![](https://img.shields.io/badge/pub-TPAMI-orange)[Keypoint recognition using randomized trees](https://ieeexplore.ieee.org/abstract/document/1661548/)

![](https://img.shields.io/badge/year-2009-g)![](https://img.shields.io/badge/pub-CVPR-orange)[From structure-from-motion point clouds to fast location recognition](https://ieeexplore.ieee.org/abstract/document/5206587/)

![](https://img.shields.io/badge/year-2012-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Worldwide pose estimation using 3d point clouds](https://link.springer.com/chapter/10.1007/978-3-319-25781-5_8)

![](https://img.shields.io/badge/year-2014-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Discriminative feature-to-point matching in image-based localization](http://openaccess.thecvf.com/content_cvpr_2014/html/Donoser_Discriminative_Feature-to-Point_Matching_2014_CVPR_paper.html)

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Hyperpoints and fine vocabularies for large-scale location recognition](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Sattler_Hyperpoints_and_Fine_ICCV_2015_paper.html)

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Efficient monocular pose estimation for complex 3d models](https://ieeexplore.ieee.org/abstract/document/7139372/)

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-TIP-orange)[Fast localization in large-scale environments using supervised indexing of binary features](https://ieeexplore.ieee.org/abstract/document/7327197/)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-BMVC-orange)[Filtering 3d keypoints using gist for accurate image-based localization](http://www.bmva.org/bmvc/2016/papers/paper127/paper127.pdf)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Large-scale location recognition and the geometric burstiness problem](http://openaccess.thecvf.com/content_cvpr_2016/html/Sattler_Large-Scale_Location_Recognition_CVPR_2016_paper.html) |[code](https://github.com/tsattler/geometric_burstiness)![](https://img.shields.io/github/stars/tsattler/geometric_burstiness?style=social)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-CVPR-orange)[From coarse to fine: robust hierarchical localization at large scale](http://openaccess.thecvf.com/content_CVPR_2019/html/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.html) |[code](https://github.com/cvg/Hierarchical-Localization)![](https://img.shields.io/github/stars/cvg/Hierarchical-Localization?style=social)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICRA-orange)[2d3d-matchnet: Learning to match keypoints across 2d image and 3d point cloud](https://ieeexplore.ieee.org/abstract/document/8794415/)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-AAAI-orange)[Lcd: Learned cross-domain descriptors for 2d-3d matching](https://ojs.aaai.org/index.php/AAAI/article/view/6859) |[code](https://github.com/hkust-vgd/lcd)![](https://img.shields.io/github/stars/hkust-vgd/lcd?style=social)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-ECCV-orange)[DA4AD: end-to-end deep attention-based visual localization for autonomous driving](https://link.springer.com/chapter/10.1007/978-3-030-58604-1_17)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-IROS-orange)[Monocular camera localization in prior lidar maps with 2d-3d line correspondences](https://ieeexplore.ieee.org/abstract/document/9341690/) |[code](https://github.com/levenberg/2D-3D-pose-tracking)![](https://img.shields.io/github/stars/levenberg/2D-3D-pose-tracking?style=social)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Back to the feature: learning robust camera localization from pixels to pose](http://openaccess.thecvf.com/content/CVPR2021/html/Sarlin_Back_to_the_Feature_Learning_Robust_Camera_Localization_From_Pixels_CVPR_2021_paper.html) |[code](https://github.com/cvg/pixloc)![](https://img.shields.io/github/stars/cvg/pixloc?style=social)

#### 2D-3D pose estimation

![](https://img.shields.io/badge/year-1981-g)![](https://img.shields.io/badge/pub-Commun._ACM-orange)[Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography](https://dl.acm.org/doi/abs/10.1145/358669.358692)

![](https://img.shields.io/badge/year-1999-g)![](https://img.shields.io/badge/pub-IWVA-orange)[Bundle adjustment — a modern synthesis](https://link.springer.com/chapter/10.1007/3-540-44480-7_21)

![](https://img.shields.io/badge/year-2000-g)![](https://img.shields.io/badge/pub-CVIU-orange)[MLESAC: a new robust estimator with application to estimating image geometry](https://www.sciencedirect.com/science/article/pii/S1077314299908329)

![](https://img.shields.io/badge/year-2003-g)![](https://img.shields.io/badge/pub-TPAMI-orange)[Complete solution classification for the perspective-three-point problem](https://ieeexplore.ieee.org/abstract/document/1217599/)

![](https://img.shields.io/badge/year-2005-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Matching with prosac — progressive sample consensus](https://ieeexplore.ieee.org/abstract/document/1467271/)

![](https://img.shields.io/badge/year-2008-g)![](https://img.shields.io/badge/pub-CVPR-orange)[ A general solution to the p4p problem for camera with unknown focal length](https://ieeexplore.ieee.org/abstract/document/4587793/)

![](https://img.shields.io/badge/year-2009-g)![](https://img.shields.io/badge/pub-IJCV-orange)[EPnP: an accurate o(n) solution to the pnp problem](https://link.springer.com/article/10.1007/s11263-008-0152-6)

![](https://img.shields.io/badge/year-2013-g)![](https://img.shields.io/badge/pub-TPAMI-orange)[USAC: a universal framework for random sample consensus](https://ieeexplore.ieee.org/abstract/document/6365642/)

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-Remote_Sens.-orange)[Direct linear transformation from comparator coordinates into object space coordinates in close-range photogrammetry](https://www.sciencedirect.com/science/article/pii/S0099111215303086)

### 3D $\rightarrow$ 2D: projection methods

![](https://img.shields.io/badge/year-2009-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Automatic registration of lidar and optical images of urban scenes](https://ieeexplore.ieee.org/abstract/document/5206539/)

![](https://img.shields.io/badge/year-2012-g)![](https://img.shields.io/badge/pub-ICRA-orange)[LAPS - localisation using appearance of prior structure: 6-dof monocular camera localisation using prior pointclouds](https://ieeexplore.ieee.org/abstract/document/6224750/)

![](https://img.shields.io/badge/year-2014-g)![](https://img.shields.io/badge/pub-IROS-orange)[Visual localization within lidar maps for automated urban driving](https://ieeexplore.ieee.org/abstract/document/6942558/)

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-BMVC-orange)[Robust direct visual localisation using normalised information distance](https://www.robots.ox.ac.uk/~mobile/Papers/2015BMVC_Pascoe.pdf)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-IROS-orange)[Sampling-based methods for visual navigation in 3d maps by synthesizing depth images](https://ieeexplore.ieee.org/abstract/document/8206067/)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-IJARS-orange)[Convolutional neural network-based coarse initial position estimation of a monocular camera in large-scale 3D light detection and ranging maps](https://journals.sagepub.com/doi/pdf/10.1177/1729881419893518)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ITSC-orange)[CMRNet: camera to lidar-map registration](https://ieeexplore.ieee.org/abstract/document/8917470/) |[code](https://github.com/cattaneod/CMRNet)![](https://img.shields.io/github/stars/cattaneod/CMRNet?style=social)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Global visual localization in lidar-maps through shared 2d-3d embedding space](https://ieeexplore.ieee.org/abstract/document/9196859/)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ECCV-orange)[CPO: change robust panorama to point cloud localization](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_11)

### 2D $\rightarrow$ 3D: Scene Dimensional Upgrading

#### Scene Coordinate Regression

![](https://img.shields.io/badge/year-2013-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Scene coordinate regression forests for camera relocalization in rgb-d images](http://openaccess.thecvf.com/content_cvpr_2013/html/Shotton_Scene_Coordinate_Regression_2013_CVPR_paper.html)

![](https://img.shields.io/badge/year-2014-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Multi-output learning for camera relocalization](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Guzman-Rivera_Multi-Output_Learning_for_2014_CVPR_paper.html)

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Exploiting uncertainty in regression forests for accurate camera relocalization](http://openaccess.thecvf.com/content_cvpr_2015/html/Valentin_Exploiting_Uncertainty_in_2015_CVPR_paper.html)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Deep image retrieval: Learning global representations for image search](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_15)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-CVPR-orange)[On-the-fly adaptation of regression forests for online camera relocalisation](http://openaccess.thecvf.com/content_cvpr_2017/html/Cavallari_On-The-Fly_Adaptation_of_CVPR_2017_paper.html)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-CVPR-orange)[DSAC — differentiable ransac for camera localization](http://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html) |[code](https://github.com/cvlab-dresden/DSAC)![](https://img.shields.io/github/stars/cvlab-dresden/DSAC?style=social)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Random forests versus neural networks — what’s best for camera localization](https://ieeexplore.ieee.org/abstract/document/7989598/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Learning less is more - 6d camera localization via 3d surface regression](http://openaccess.thecvf.com/content_cvpr_2018/html/Brachmann_Learning_Less_Is_CVPR_2018_paper.html) |[code](https://github.com/vislearn/LessMore)![](https://img.shields.io/github/stars/vislearn/LessMore?style=social)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-ECCVW-orange)[Scene coordinate regression with angle-based reprojection loss for camera relocalization](https://openaccess.thecvf.com/content_eccv_2018_workshops/w16/html/Li_Scene_Coordinate_Regression_with_Angle-Based_Reprojection_Loss_for_Camera_Relocalization_ECCVW_2018_paper.html?ref=https://githubhelp.com)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-IROS-orange)[Exploiting points and lines in regression forests for rgb-d camera relocalization](https://ieeexplore.ieee.org/abstract/document/8593505/)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Expert sample consensus applied to camera re-localization](http://openaccess.thecvf.com/content_ICCV_2019/html/Brachmann_Expert_Sample_Consensus_Applied_to_Camera_Re-Localization_ICCV_2019_paper.html) |[code](https://github.com/vislearn/esac)![](https://img.shields.io/github/stars/vislearn/esac?style=social)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICCV-orange)[SANet: scene agnostic network for camera localization](http://openaccess.thecvf.com/content_ICCV_2019/html/Yang_SANet_Scene_Agnostic_Network_for_Camera_Localization_ICCV_2019_paper.html)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICCVW-orange)[ Camera relocalization by exploiting multi-view constraints for scene coordinates regression](http://openaccess.thecvf.com/content_ICCVW_2019/html/DL4VSLAM/Cai_Camera_Relocalization_by_Exploiting_Multi-View_Constraints_for_Scene_Coordinates_Regression_ICCVW_2019_paper.html)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Hierarchical scene coordinate classification and regression for visual localization](http://openaccess.thecvf.com/content_CVPR_2020/html/Li_Hierarchical_Scene_Coordinate_Classification_and_Regression_for_Visual_Localization_CVPR_2020_paper.html)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-CVPR-orange)[ KFNet: learning temporal camera relocalization using kalman filtering](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_KFNet_Learning_Temporal_Camera_Relocalization_Using_Kalman_Filtering_CVPR_2020_paper.html) |[code](https://github.com/zlthinker/KFNet)![](https://img.shields.io/github/stars/zlthinker/KFNet?style=social)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Learning camera localization via dense scene matching](http://openaccess.thecvf.com/content/CVPR2021/html/Tang_Learning_Camera_Localization_via_Dense_Scene_Matching_CVPR_2021_paper.html) |[code](https://github.com/Tangshitao/Dense-Scene-Matching)![](https://img.shields.io/github/stars/Tangshitao/Dense-Scene-Matching?style=social)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-TPAMI-orange)[Visual camera re-localization from RGB and RGB-D images using DSAC](https://ieeexplore.ieee.org/abstract/document/9394752/) |[code](https://github.com/vislearn/dsacstar)![](https://img.shields.io/github/stars/vislearn/dsacstar?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Learning to detect scene landmarks for camera localization](http://openaccess.thecvf.com/content/CVPR2022/html/Do_Learning_To_Detect_Scene_Landmarks_for_Camera_Localization_CVPR_2022_paper.html)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-RAL-orange)[A deep feature aggregation network for accurate indoor camera localization](https://ieeexplore.ieee.org/abstract/document/9697338/)

#### point cloud Reconstruction

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-IROS-orange)[Monocular camera localization in 3d lidar maps](https://ieeexplore.ieee.org/abstract/document/7759304/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-IROS-orange)[Stereo camera localization in 3d lidar maps](https://ieeexplore.ieee.org/abstract/document/8594362/)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-CAVW-orange)[Scale-aware camera localization in 3d lidar maps with a monocular visual odometry](https://onlinelibrary.wiley.com/doi/abs/10.1002/cav.1879)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-ISPRS-orange)[3D map-guided single indoor image localization refinement](https://www.sciencedirect.com/science/article/pii/S0924271620300083)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-CVPR-orange)[DeepI2P: image-to-point cloud registration via deep classification](http://openaccess.thecvf.com/content/CVPR2021/html/Li_DeepI2P_Image-to-Point_Cloud_Registration_via_Deep_Classification_CVPR_2021_paper.html) |[code](https://github.com/lijx10/DeepI2P)![](https://img.shields.io/github/stars/lijx10/DeepI2P?style=social)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-Remote_Sens.-orange)[Autonomous vehicle localization with prior visual point cloud map constraints in gnss-challenged environments]()

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-RAL-orange)[Mobile robot localization considering uncertainty of depth regression from camera images](https://ieeexplore.ieee.org/abstract/document/9669107/)

## Visual relocalization in Dense Boundary Representation Maps

### Mesh map

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-ICRA-orange)[FARLAP: fast robust localisation using appearance priors](https://ieeexplore.ieee.org/abstract/document/7140093/)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-ECMR-orange)[Monocular localization in feature-annotated 3d polygon maps](https://ieeexplore.ieee.org/abstract/document/9568810/)

### Surfel Map

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Monocular direct sparse localization in a prior 3d surfel map](https://ieeexplore.ieee.org/abstract/document/9197022/)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-ICRA-orange)[3D surfel map-aided visual relocalization with learned descriptors](https://ieeexplore.ieee.org/abstract/document/9561005/)

### SDF Map

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-IROS-orange)[Metric monocular localization using signed distance fields](https://ieeexplore.ieee.org/abstract/document/8968033/)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-RAL-orange)[ Freetures: localization in signed distance function map](https://ieeexplore.ieee.org/abstract/document/9327493/)

## Visual relocalization in high definition(HD) Map

![](https://img.shields.io/badge/year-2013-g)![](https://img.shields.io/badge/pub-IROS-orange)[Light-weight localization for vehicles using road markings](https://ieeexplore.ieee.org/abstract/document/6696460/)

![](https://img.shields.io/badge/year-2013-g)![](https://img.shields.io/badge/pub-IV-orange)[LaneLoc: lane marking based localization using highly accurate maps](https://ieeexplore.ieee.org/abstract/document/6629509/)

![](https://img.shields.io/badge/year-2014-g)![](https://img.shields.io/badge/pub-IV-orange)[Monocular visual localization using road structural features](https://ieeexplore.ieee.org/abstract/document/6856539/)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-IROS-orange)[Pole-based localization for autonomous vehicles in urban scenarios](https://ieeexplore.ieee.org/abstract/document/7759339/)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-IV-orange)[ Improving vehicle localization using semantic and pole-like landmarks](https://ieeexplore.ieee.org/abstract/document/7995692/)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-IV-orange)[Monocular localization in urban environments using road markings](https://ieeexplore.ieee.org/abstract/document/7995762/)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-IROS-orange)[Monocular localization in hd maps by combining semantic segmentation and distance transform](https://ieeexplore.ieee.org/abstract/document/9341003/)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-JAT-orange)[In-lane localization and ego-lane identification method based on highway lane endpoints](https://www.hindawi.com/journals/jat/2020/8684912/)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-Sensors-orange)[ Monocular localization with vector hd map (mlvhm): a low-cost method for commercial ivs](https://www.mdpi.com/1424-8220/20/7/1870)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-IROS-orange)[Coarse-to-fine semantic localization with hd map for autonomous driving in structural scenes](https://ieeexplore.ieee.org/abstract/document/9635923/)

![](https://img.shields.io/badge/year-2021-g)![](https://img.shields.io/badge/pub-IROS-orange)[BSP-monoloc: basic semantic primitives based monocular localization on roads](https://ieeexplore.ieee.org/abstract/document/9636321/)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ICRA-orange)[LTSR: long-term semantic relocalization based on hd map for autonomous vehicles](https://ieeexplore.ieee.org/abstract/document/9811855/)

## Visual relocalization in Semantic Maps

### Semantic Global  Features

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Lost shopping! monocular localization in large indoor spaces](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Wang_Lost_Shopping_Monocular_ICCV_2015_paper.html)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Do you see the bakery? leveraging geo-referenced texts for global localization in public maps](https://ieeexplore.ieee.org/abstract/document/7487688/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Semantic visual localization](http://openaccess.thecvf.com/content_cvpr_2018/html/Schonberger_Semantic_Visual_Localization_CVPR_2018_paper.html)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-CVPR-orange)[DeLS-3d: deep localization and segmentation with a 3d semantic map](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_DeLS-3D_Deep_Localization_CVPR_2018_paper.html) |[code](https://github.com/pengwangucla/DeLS-3D)![](https://img.shields.io/github/stars/pengwangucla/DeLS-3D?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-CVPRW-orange)[Semantic pose verification for outdoor visual localization with self-supervised contrastive learning](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/html/Orhan_Semantic_Pose_Verification_for_Outdoor_Visual_Localization_With_Self-Supervised_Contrastive_CVPRW_2022_paper.html)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ESTIJ-orange)[Long-term image-based vehicle localization improved with learnt semantic descriptors](https://www.sciencedirect.com/science/article/pii/S2215098622000064)

### Semantic Feature matching

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Semantic match consistency for long-term visual localization](http://openaccess.thecvf.com/content_ECCV_2018/html/Carl_Toft_Semantic_Match_Consistency_ECCV_2018_paper.html)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Long-term visual localization using semantically segmented images](https://ieeexplore.ieee.org/abstract/document/8463150/)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Fine-grained segmentation networks: self-supervised segmentation for improved long-term visual localization](http://openaccess.thecvf.com/content_ICCV_2019/html/Larsson_Fine-Grained_Segmentation_Networks_Self-Supervised_Segmentation_for_Improved_Long-Term_Visual_Localization_ICCV_2019_paper.html) |[code](https://github.com/maunzzz/fine-grained-segmentation-networks)![](https://img.shields.io/github/stars/maunzzz/fine-grained-segmentation-networks?style=social)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ICRA-orange)[SemLoc: accurate and robust visual localization with semantic and structural constraints from prior maps](https://ieeexplore.ieee.org/abstract/document/9811925/)

### Robust Feature Selection

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Semantically guided location recognition for outdoors scenes](https://ieeexplore.ieee.org/abstract/document/7139877/)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICRA-orange)[Semantics-aware visual localization under challenging perceptual conditions](https://ieeexplore.ieee.org/abstract/document/7989305/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-IROS-orange)[VLASE: vehicle localization by aggregating semantic edges](https://ieeexplore.ieee.org/abstract/document/8594358/) |[code](https://github.com/sagachat/VLASE)![](https://img.shields.io/github/stars/sagachat/VLASE?style=social)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-BMVC-orange)[Semantically-aware attentive neural embeddings for 2d long-term visual localization](https://bmvc2019.org/wp-content/uploads/papers/0443-paper.pdf)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ICRA-orange)[PixSelect: less but reliable pixels for accurate and efficient localization](https://ieeexplore.ieee.org/abstract/document/9812345/)

## Visual relocalization in NeRF map

![](https://img.shields.io/badge/year-2023-g)![](https://img.shields.io/badge/pub-ICRA-orange)[NeRF-Loc: Visual Localization with Conditional Neural Radiance Field](https://arxiv.org/abs/2304.07979)

## Dataset

### Outdoor

![](https://img.shields.io/badge/year-2010-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Accurate image localization based on google maps street view](https://link.springer.com/chapter/10.1007/978-3-642-15561-1_19)

![](https://img.shields.io/badge/year-2011-g)![](https://img.shields.io/badge/pub-CVPR-orange)[San francisco](https://ieeexplore.ieee.org/abstract/document/5995610/) |[page](https://sites.google.com/site/chenmodavid/datasets)

![](https://img.shields.io/badge/year-2012-g)![](https://img.shields.io/badge/pub-ECCV-orange)[Dubrovnik 6K](https://link.springer.com/chapter/10.1007/978-3-319-25781-5_8) |[page](https://mldta.com/dataset/dubrovnik6k-and-rome16k/)

![](https://img.shields.io/badge/year-2013-g)![](https://img.shields.io/badge/pub-IJRR-orange)[KITTI](https://journals.sagepub.com/doi/pdf/10.1177/0278364913491297) |[page](https://www.cvlibs.net/datasets/kitti/)

![](https://img.shields.io/badge/year-2015-g)![](https://img.shields.io/badge/pub-ICCV-orange)[Cambridge](http://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html) |[page](https://www.repository.cam.ac.uk/handle/1810/251342)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Cityscape](http://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html) |[page](https://www.cityscapes-dataset.com/dataset-overview/)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-IJRR-orange)[NCLT](https://journals.sagepub.com/doi/pdf/10.1177/0278364915614638) |[page](http://robots.engin.umich.edu/nclt/)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-IJRR-orange)[Oxford-robotcar](https://journals.sagepub.com/doi/pdf/10.1177/0278364916679498) |[page](http://robotcar-dataset.robots.ox.ac.uk/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-CVPR-orange)[Aachen Day-Night](http://openaccess.thecvf.com/content_cvpr_2018/html/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.html) |[page](http://visuallocalization.net/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-CVPR-orange)[CMU Seasons](http://openaccess.thecvf.com/content_cvpr_2018/html/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.html) |[page](https://www.visuallocalization.net/datasets/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-CVPRW-orange)[Apollo Scape](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w14/html/Huang_The_ApolloScape_Dataset_CVPR_2018_paper.html) |[page](https://apolloscape.auto/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-TITS-orange)[KAIST Day/Night](https://ieeexplore.ieee.org/abstract/document/8293689/) |[page](https://github.com/sejong-rcv/CVPRW2015.PlaceRecognition)

![](https://img.shields.io/badge/year-2019-g)![](https://img.shields.io/badge/pub-ICCV-orange)[SemanticKITTI](http://openaccess.thecvf.com/content_ICCV_2019/html/Behley_SemanticKITTI_A_Dataset_for_Semantic_Scene_Understanding_of_LiDAR_Sequences_ICCV_2019_paper.html) |[page](http://www.semantic-kitti.org/)

![](https://img.shields.io/badge/year-2020-g)![](https://img.shields.io/badge/pub-CVPR-orange)[nuScenes](http://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html) |[page](https://www.nuscenes.org/)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-CVPR-orange)[SF-XL](http://openaccess.thecvf.com/content/CVPR2022/html/Berton_Rethinking_Visual_Geo-Localization_for_Large-Scale_Applications_CVPR_2022_paper.html) |[page](https://github.com/gmberton/cosplace)

### Indoor

![](https://img.shields.io/badge/year-2012-g)![](https://img.shields.io/badge/pub-IROS-orange)[TUM-RGBD](https://ieeexplore.ieee.org/abstract/document/6385773/) |[page](https://vision.in.tum.de/data/datasets/rgbd-dataset)

![](https://img.shields.io/badge/year-2013-g)![](https://img.shields.io/badge/pub-CVPR-orange)[7 scenes](http://openaccess.thecvf.com/content_cvpr_2013/html/Shotton_Scene_Coordinate_Regression_2013_CVPR_paper.html) |[page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-3DV-orange)[SceneNN](https://ieeexplore.ieee.org/abstract/document/7785081/) |[page](http://103.24.77.34/scenenn/home/)

![](https://img.shields.io/badge/year-2016-g)![](https://img.shields.io/badge/pub-IJRR-orange)[EuRoC MAV](https://journals.sagepub.com/doi/pdf/10.1177/0278364915620033) |[page](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-CVPR-orange)[3DMatch](http://openaccess.thecvf.com/content_cvpr_2017/html/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.html) |[page](http://3dmatch.cs.princeton.edu/)

![](https://img.shields.io/badge/year-2018-g)![](https://img.shields.io/badge/pub-ECCV-orange)[ADVIO](http://openaccess.thecvf.com/content_ECCV_2018/html/Santiago_Cortes_ADVIO_An_Authentic_ECCV_2018_paper.html) |[page](https://github.com/AaltoVision/ADVIO)

### Misc

![](https://img.shields.io/badge/year-2017-g)![](https://img.shields.io/badge/pub-ICCV-orange)[SceneNet RGB-D](http://openaccess.thecvf.com/content_iccv_2017/html/McCormac_SceneNet_RGB-D_Can_ICCV_2017_paper.html) |[page](https://robotvault.bitbucket.io/scenenet-rgbd.html)

![](https://img.shields.io/badge/year-2022-g)![](https://img.shields.io/badge/pub-ECCV-orange)[LaMAR](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_40) |[page](https://lamar.ethz.ch/)

