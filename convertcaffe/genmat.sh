base=/home/zhizhong/Dataset/alexnet_places2
# please change deploy_edited.prototxt to pre-2015 caffe format, i.e. replace "Convolution" with CONVOLUTION, etc.
python2 import-caffe.py \
    --caffe-variant=caffe \
    --preproc=caffe \
    --average-value="(116.282836914, 114.063842773, 105.908874512)" \
    "$base/deploy_edited.prototxt" \
    "$base/alexnet_places2.caffemodel" \
    "$base/alexnet_places2.mat"
