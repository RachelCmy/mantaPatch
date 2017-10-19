MantaPatch
============

This repo contains the code of the paper:

[Data-driven Synthesis of Smoke Flows with Cnn-based Feature Descriptors, Mengyu Chu and Nils Thuerey](https://dl.acm.org/citation.cfm?id=3073643).

Install
---------------
- Install CMake, python, tensorflow (numpy), matplot, sklearn, flann(more details in the very last section)
- Clone this repo
  ```
  git clone https://github.com/RachelCmy/mantaPatch.git
  ```
- The code is implemented on mantaflow. 
  Here is an installation guide: http://mantaflow.com/install.html
- When building mantaflow with CMake, it is necessary to check ```NUMPY``` on.
  It is recommended to check the ```GUI``` (depends on Qt) and ```OPENMP``` on as well.
- eigen(http://eigen.tuxfamily.org/index.php?title=Main_Page) is used in our code. 
  We have one under ```dependencies``` folder, and it is used by default on Windows.
  On Linux, please edit the eigen path in ```CMakeLists.txt``` file, at line 481
- Related scene files are in folder ```mantaflow_path/tensorflow/example3_smoke_patch/```
  or folder ```mantaflow_path/scenes/```
- Output files are usually in folder ```mantaflow_path/tensorflow/data```
- Note the working folder used below is always ```mantaflow_path/build/```
- The ```manta``` below stands for the executable file. 
    * On Windows, it is ```Release/manta.exe``` (mantaflow_path/build/Release/manta.exe)
    * On Linux, it is ```./manta``` (mantaflow_path/build/manta)

Try Patch Advection
---------------
Examples of deformation-limiting patch advection scenes in 3d and 2d:
```
manta ../scenes/patch_2d.py
# or
manta ../scenes/patch_3d.py
```
2d is visualized with smile faces and 3d is visualized with meshes( try shift+m ).


Train a patch-descriptor-learning CNN
---------------
1. A Simulation Pair Generation:
    
        # for 2D
        manta ../tensorflow/example3_smoke_patch/genSimData.py
        # or for 3D
        manta ../tensorflow/example3_smoke_patch/genSimData.py dim 3
        
    It will generate one simulation pair.
    All data is saved in ```../tensorflow/data/sim_????/``` folder.
    3D simulations will take longer time.
    It is necessary to **generate lots of pairs** (with different seeds) to get good training results.
    
    Examples for parameters:
    
        # example1, input a single number, as random seed,
        # for e.g. the 42 will be used as random seed here.
        manta ../tensorflow/example3_smoke_patch/genSimData.py 42 
        # example2, input a parameter-list. 
        # Only right names are accepted
        # ( take a look at line 252 in genSimData.py)
        manta ../tensorflow/example3_smoke_patch/genSimData.py npSeed 42 resetN 20 saveppm False
    
2. Patch Pairs Generation (Training Data for step 3):
    
        manta ../tensorflow/example3_smoke_patch/genPatchData.py 1000
    
    This command will generate the patch pairs from the simulation in ```data/sim_1000/```
    All data is saved in ```../tensorflow/data/patch_1000/``` folder.
    1000 can be replaced with any other valid ```simNo```.
    A special case is ```simNo == -1```
    
        manta ../tensorflow/example3_smoke_patch/genPatchData.py -1
        # or
        manta ../tensorflow/example3_smoke_patch/genPatchData.py simNo -1 savePatchPpm True
    
    This command will generate patch pairs from all existed simulations in folders named as ```data/sim_????/```,
    except for the ones whose patch folders already exist.
    
    Examples for parameter-list:
    
        # Only right names are accepted
        #( take a look at line 128 in genSimData.py)
        manta ../tensorflow/example3_smoke_patch/genPatchData.py \
        simNo 1002 PATCH_SIZE 48 savePatchPpm False
    
    Note that the factor between coase and fine are already defined in the simulation pair. 
    ```PATCH_SIZE``` is used as the size for high resolution patches, and for coarse ones the size is ```PATCH_SIZE/factor```.
    And 36 (by default) is recommanded.
    
    If ```savePatchPpm``` is true, ppm image files will be outputed with patch uni files. 
    They are very large ( no image compression used ), but can help to visualize patches, 
    with the software djv_view(http://djv.sourceforge.net/).
    
3. Train a Saimese CNN:
    Parameter-list is at line 253 in trainPatchData.py.
    Parameter ```data_dirs``` stands for the training dataset pathes.
    It should be in a pair of quotes, multiple pathes should be seperated with commas, for e.g.,
    
        # train a density descriptor first
        manta ../tensorflow/example3_smoke_patch/trainPatchData.py \
        data_dirs "../tensorflow/data/patch_1000/, ../tensorflow/data/patch_1001/"
        # train a curl-of-velocity descriptor as well
        manta ../tensorflow/example3_smoke_patch/trainPatchData.py \
        data_dirs "../tensorflow/data/patch_1000/, ../tensorflow/data/patch_1001/" dataFlag 2
        # make sure that data_dirs path strings end with "/" or "\\" !
    
    Training log, trained models and other files are saved in ```data/_trainlog_yyyymmdd_hhmmss/``` folder
    
4. Evaluate the CNN:
    Parameter-list is at line 207 in RecallEval.py.
    
    For a given evaluation dataset( should be different from the training one), 
    we evaluate the given trained model with the recall over rank curve.
    
        # evaluate a density descriptor (dataFlag = 1)
        manta ../tensorflow/example3_smoke_patch/RecallEval.py \
        data_dirs "../tensorflow/data/patch_1002/" \
        saved_model ../tensorflow/data/_trainlog_yyyymmdd_hhmmss/model_yyyymmdd_hhmmss.ckpt \
        dataFlag 1
        # evaluate a curl descriptor (dataFlag = 2)
        manta ../tensorflow/example3_smoke_patch/RecallEval.py \
        data_dirs "../tensorflow/data/patch_1002/" \
        saved_model ../tensorflow/data/_trainlog_yyyymmdd_hhmmss/model_yyyymmdd_hhmmss.ckpt \
        dataFlag 2
        # param color could be used to set the color of the recall curve
        # make sure that dataFlag and the saved_model are consistent
        # (both for density or both for curl, and both for 2D or 3D).
    
    Evaluation log, recall curve, descriptors and other files 
    are saved in ```data/_evaltlog_yyyymmdd_hhmmss/``` folder.
    For parameter ```saved_model```, our trained models are offered in folder ```mantaflow_path/tensorflow/example3_smoke_patch/models/```
    Note here, unlike the training code, there is no automatic dataset separation, so a relatively smaller evaluation dataset is supposed to be used.
    
Prepare a repository
---------------
1. High-resolution Simulation Generation Only:
    It is possible to reuse the ```training/evaluation``` dataset, 
    but it is better to create some even larger-resolution simulations, for e.g.,
    
        manta ../tensorflow/example3_smoke_patch/genSimData.py doCoarse False res 128 simNo 2000
    
    You can also use specially designed simulations ( a special simNo should be used ), for e.g.,
    
        # for 2d, the only "special" part is that the random seeds are fixed ( at line 62 )
        manta ../tensorflow/example3_smoke_patch/genSimDataSpecial.py
        # or for 3d, the only "special" part is that the random seeds are fixed ( at line 62 )
        manta ../tensorflow/example3_smoke_patch/genSimDataSpecial.py dim 3 
        # high-res in 3D is very slow
    
    The output files should have the same folder structure.
    Then generate larger patches as well (PATCH_SIZE = 64, 72 or even larger):
    
        manta ../tensorflow/example3_smoke_patch/genPatchData.py \
        doCoarse False simNo 2000 PATCH_SIZE 64
    
2. Use a trained CNN model to extract all descriptors:
    
        # density des
        # note that PATCH_SIZE should be the same as the trained CNN
        # instead of the repository patch size.
        manta ../tensorflow/example3_smoke_patch/libDesPack.py dataFlag 1 PATCH_SIZE 36 
        saved_model ../tensorflow/example3_smoke_patch/models/model_2D/den_cnn.ckpt 
        data_dirs "../tensorflow/data/patch_2000/"
        
        # curl of velocity des
        manta ../tensorflow/example3_smoke_patch/libDesPack.py dataFlag 2 PATCH_SIZE 36 
        saved_model ../tensorflow/example3_smoke_patch/models/model_2D/curl_cnn.ckpt 
        data_dirs "../tensorflow/data/patch_2000/"
        
        # or use some other model:
        # saved_model ../tensorflow/data/_trainlog_yyyymmdd_hhmmss/model_yyyymmdd_hhmmss.ckpt
        # set DIM as 3 to work with 3D repository
    
    
    Output descriptor npz file will be saved in ```data/_deslog_yyyymmdd_hhmmss/``` folder, 
    unless parameter ```desdataPath``` is assigned as other output directory.
    If the parameter ```updateLib``` is ```True``` (as default), the descriptor file will be 
    copied to the repository folder as well.
    
Get new high-resolution simulations
---------------
1. A 2D example of smoke synthesis with repository patches using CNN descriptors.
    
        manta ../tensorflow/example3_smoke_patch/patch_2d.py \
        SynCurlFlag True data_dirs "../tensorflow/data/patch_2000/" \
        den_model ../tensorflow/data/_trainlog_yyyymmdd_hhmmss/model_yyyymmdd_hhmmss.ckpt \
        curl_model ../tensorflow/data/_trainlog_yyyymmdd_hhmmss/model_yyyymmdd_hhmmss.ckpt
        # den_model & curl_model, our trained models are used by default (if not given)
    
    An extra library, flann, is used to build the kd-tree of the repository.
    It can be installed with pip, more info: https://github.com/primetang/pyflann
    
    For python3, it is necessary to fix some problems, according to:
    https://github.com/primetang/pyflann/issues/1
    It is important to use the same model as used in the repository (See Prepare a repository)
    Our repository patches (using our models) are not uploaded with the code.
    But the 2D version can be downloaded from: https://ge.in.tum.de/download/data/mantaPatchData.7z
    
        wget https://ge.in.tum.de/download/data/mantaPatchData.7z
        7z x mantaPatchData.7z -o../tensorflow/example3_smoke_patch/
        # or on windows, 64 bit (install https://eternallybored.org/misc/wget/)
        wget64 https://ge.in.tum.de/download/data/mantaPatchData.7z
        "path/to/7z.exe" x mantaPatchData.7z -o../tensorflow/example3_smoke_patch/
        # then you can run the patch_2d.py with all default parameters
        manta ../tensorflow/example3_smoke_patch/patch_2d.py
    Output simulations are saved in folder ```../tensorflow/data/_simlog_yyyymmdd_hhmmss/```,
    and ppm files are saved there for visualization.
    
2. A 3D example:
    
        manta ../tensorflow/example3_smoke_patch/patch_3d.py \
        data_dirs "../tensorflow/data/patch_7001/" \
        den_model ../tensorflow/example3_smoke_patch/models/model_3D/den_cnn.ckpt \
        curl_model ../tensorflow/example3_smoke_patch/models/model_3D/curl_cnn.ckpt
    
    
    In this code, we still load all the patches and do the volume synthesis frame by frame.
    Additionally, we save all the high-res output uni files in to output folder.
    That's why this is still slow. As written in the paper, one can also only save all patch
    info to the rendering scene file, and don't high-res patches until rendering when every 
    frame could work in parallel. In that way, it is more effective.
    The rendering part (as well as the generation of rendering scene files) is not included 
    with this code yet.
    
Acknowledgements
---------------
This work was funded by the ERC Starting Grant realFlow (ERC StG-2015-637014).