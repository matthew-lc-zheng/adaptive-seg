# In-situ adaptation for the task of instance segmentation
The code in *adaptive-seg* links the dissertation of ***Domain adaptation for instance segmentation***, 2020.
The part of transfer masks and boxes is shown and published here. The codes of extra algorithms in the thesis is forthcoming and will be opened in future.
For further consultation or a request of this code and future code, please contact with *matthew.lc.zheng@outlook.com* or *matthew.lc.zheng@protonmail.com* officially. All informal conversations are suggested being done by *zsms123zlc@gmail.com* or *libraneptune@163.com*

## Installation and requirements
package list in Python(Py3.x version):
```
pip install torch==1.4.0+cu100 random argparse opencv-python numpy
```
This code is based on the release of detectron2 from FAIR. That means detectron2 is supposed to be installed as well. For the detail of setup, please refer to the project of detectron2.
By the way, for the stability of the code, keep the version of  pytorch fixed and same with the version introduced here. Other versions are not guaranteed to work properly.

## File organization
As for `run.py`
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
As for `atomizer.py`
```
root
|——input
|——output
     |——result
```
## Usage
Basic usage for `run.py` is as follows:
```
cd path to run.py
python run.py --input_A path_to_dataset_after_adaptation --input_T path_to_target_dataset --output path_to_save
```
Basic usage for `atomizer.py` is as follows:
```
cd path to atomizer.py
python main.py --input ./input --output ./output 
```
Above example shows the production of fog defaulty and if you hope a mog, please set the coefficient `--transpancy` in the range of `[0.1,0.4]` while `[0.6,0.9]` works for fog.
More options of the usage can be find in the next section.

**Extra**: For the operation of atomization, the code in cpp runs at least 100x faster than in python. You are permitted to set corresponding parameters in the cpp file to run it with less time consumpation.

## Options
More options are available in --help
```
python run.py --help
python atomizer.py --help

```
