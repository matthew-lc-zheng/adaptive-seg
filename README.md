# In-situ adaptation for the task of instance segmentation
The code in *adaptive-seg* links the dissertation of ***Domain adaptation for instance segmentation***, 2020.
The part of transfer masks and boxes is shown and published here. The codes of extra algorithms in the thesis is forthcoming and will be opened in future.
For further consultion or a request of this code and future code, please contact with *matthew.lc.zheng@outlook.com* or *matthew.lc.zheng@protonmail.com* officially. All informal conversations are suggested being done by *zsms123zlc@gmail.com* or *libraneptune@163.com*

## Installation
package list in Python(Py3.x version):
```
pip install torch==1.4.0+cu100 random argparse opencv-python numpy matplotlib
```
This code is based on the release of detectron2 from FAIR. That means detectron2 is supposed to be installed as well. For the detail of setup, please refer to the project of detectron2.
By the way, for the stability of the code, keep the version of  pytorch fixed and same with the version introduced here. Other versions are not guaranteed to work properly.

## File organization
```
root
|——adaptation
     |——mog
     |——thin
     |——thick
     |——mog-aug
     |——fog-aug
|——target
     |——mog
     |——thin
     |——thick
     |——mog-aug
     |——fog-aug
|——result
     |——mog
     |——thin
     |——thick
     |——mog-aug
     |——fog-aug
```

## Usage
Basic usage for adaptive segmentation is as follows:
```
cd path to run.py
python run.py --input_A path_to_dataset_after_adaptation --input_T path_to_target_dataset --output path_to_save
```
## API
More options are available in --help
```
python run.py --help

```
