### Deep Learning Course Freiburg, exercise to train a RGB-D network in tensorflow using a subset (10 classes) of the RGB-D object dataset from the University of Washington. <br/>
```sh
# rgb only
$ python convolutional.py
# rgb-d
$ python convolutional.py --use_rgbd 
```
### The data for this exercise is a downsampled version of the dataset from
    @inproceedings{lai2011large,
      title={A large-scale hierarchical multi-view rgb-d object dataset},
      author={Lai, Kevin and Bo, Liefeng and Ren, Xiaofeng and Fox, Dieter},
      booktitle={Robotics and Automation (ICRA), 2011 IEEE International Conference on},
      pages={1817--1824},
      year={2011},
      organization={IEEE}
    }
