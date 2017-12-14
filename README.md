# Parallel-image-segmentation

The image segmentation algorithm we use has two parts: firstly generate superpixels by SLIC algorithm and then segment the image using max-flow-min-cut.
These two parts are parallelized separately.

### To do

- [x] implementation of slic
- [x] compiling (just use nvcc)
- [x] debug slic
- [ ] implementation of max-flow-min-cut
- [ ] debug mfmc
- [ ] test result

### Reference
[SLIC](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.8269&rep=rep1&type=pdf)

[c++ implementation of SLIC](https://github.com/PSMM/SLIC-Superpixels)

[gpu implementation of SLIC](https://github.com/carlren/gSLICr)
