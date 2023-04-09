#  CycDA: Unsupervised Cycle Domain Adaptation to Learn from Image to Video 

---
Official implementation of CycDA [[arXiv](https://arxiv.org/abs/2203.16244)]
---
## Requirements
* Our experiments run on Python 3.6 and PyTorch 1.7. Other versions should work but are not tested. 
* All dependencies can be installed using pip:  
`python -m pip install -r requirements.txt
`

---
## Data Preparation
* Image Datasets
    * Stanford-40: [download](http://vision.stanford.edu/Datasets/40actions.html)
    * HII: [download](https://vision.cs.hacettepe.edu.tr/interaction_images/)
    * BU101: [download](https://cs-people.bu.edu/sbargal/BU-action/)
* Video Datasets
    * UCF101: [download](https://www.crcv.ucf.edu/data/UCF101.php)
    * HMDB51: [download](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
   
* Other required data for BU101&rarr;UCF101 can be downloaded [here](https://drive.google.com/drive/folders/1OBbcpISzgOejE4aDwt0B7PmOlN89Uv1g?usp=sharing)  
    `UCF_mapping.txt`: mapping file of `class_id action_name`  
    `list_BU2UCF_img_new.txt`: list of image files (resized) `img_path label`  
    `list_ucf_all_train_split1.txt`: list of training videos `video_path label`  
    `list_ucf_all_val_split1.txt`: list of test videos `video_path label`  
    After extracting frames of videos  
    `list_frames_train_split1.txt`: list of frames of training videos `frame_path label`  
    `list_frames_val_split1.txt`: list of frames of test videos `frame_path label`  
    
    `ucf_all_vid_info.npy`:  dictionary of {videoname: (n_frames, label)}  
    `InceptionI3d_pretrained/rgb_imagenet.pt`: I3D Inception v1 pretrained on the Kinetics dataset 
* Data structure
    ```
    DATA_PATH/
      UCF-HMDB_all/
        UCF/
          CLASS_01/
            VIDEO_0001.avi
            VIDEO_0002.avi
            ...
          CLASS_02/
          ...
    
        ucf_all_imgs/
          all/
            CLASS_01/
                VIDEO_0001/
                  img_00001.jpg
                  img_00002.jpg
                  ...
                VIDEO_0002/
              ...
       BU101/
        img_00001.jpg
        img_00002.jpg
        ...
    ```

---
## Usage
Here we release the codes for training in separate stages on BU101&rarr;UCF101. Scripts for a complete cycle are still in construction.     
* stage 1 - Class-agnostic spatial alignment
    train the image model, output frame-level pseudo labels
    ```
    python bu2ucf_train_stage1.py
    ```
* stage 3 - Class-aware spatial alignment
    train the image model with video-level pseudo labels from stage 2, output frame-level pseudo labels
    ```
    python bu2ucf_train_stage3.py --target_train_vid_ps_label ps_from_stage2.npy
    ```
* stage 2 & 4 - Spatio-temporal learning
    * video model training: train the video model with frame-level pseudo labels from the image model
        * specify in the config file:  
            * `data_dir`: main data directory
            * `pseudo_gt_dict`: frame-level pseudo labels from image model training
            * `pretrained_model_path`: path of pretrained model
            * `work_main_dir`: directory of training results and logs
            * `ps_thresh`: confidence threshold p of frame-level thresholding     
        * specify in `mmaction/utils/update_config.py`: path of `target_train_vid_info`   
        * specify the `--gpu-ids` used for training  
        ```
        ./tools/dist_train_da.sh configs/recognition/i3d/ucf101/i3d_incep_da_kinetics400_rgb_video_1x64_strid1_test3clip_128d_w_ps_img0.8_targetonly_split1.py 4 --gpu-ids 0 1 2 3 --validate
        ```
    * pseudo label computation: compute video-level pseudo labels
        ```
        model_path=your_model_path
        ps_name=epoch_20_test1clip
        python tools/test_da.py configs/recognition/i3d/ucf101/i3d_incep_da_kinetics400_rgb_video_1x64_strid1_test1clip_128d_split2_compute_ps.py $model_path --out $model_dir/$ps_name.json --eval top_k_accuracy
        ```
        
---
## Acknowledgements
Some codes are adapted from [mmaction2](https://github.com/open-mmlab/mmaction2) and [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
