addpath ./matlab
matconvnet = load('models/alexnet_places2.mat'); % computed using import-caffe.py from MatConvNet v0.13. Not sure if a newer one works better.
% For some reason I cannot get it to transfer the weights. It just transfer the structure. See genmat.sh for an example:
% python2 import-caffe.py \
%     --caffe-variant=caffe \
%     --preproc=caffe \
%     --average-value="(116.282836914, 114.063842773, 105.908874512)" \
%     "$base/deploy_edited.prototxt" \ # changing to pre-2015 caffe format, i.e. replace "Convolution" with CONVOLUTION, etc.
%     "$base/alexnet_places2.caffemodel" \
%     "$base/alexnet_places2.mat"
% Note that the deploy_edited.prototxt has to be edited to a pre-2015 caffe format to work on MatConvNet v0.13's importer.

caffenet = caffe.Net('models/alexnet_places2_deploy.prototxt', 'models/alexnet_places2.caffemodel', 'test');
matlayers = find( cellfun( @(X) isfield(X, 'weights'), matconvnet.layers ) );
caffelayers = find( ~cellfun(@isempty, {caffenet.layer_vec.params}) );
assert(numel(matlayers) == numel(caffelayers));

% transfer each conv layer
for i = 1:numel(matlayers)
    w = caffenet.layer_vec( caffelayers(i) ).params(1).get_data();
    % difference for the first fc layer: as fc layer in caffe, but h,w~=1 in matconvnet
    sizematw = size(matconvnet.layers{ matlayers(i) }.weights{1});
    if ( ndims(w)~= numel(sizematw) ) 
        w = reshape(w, sizematw(1), sizematw(2), [], sizematw(4));
    end
    % other differences
    if ndims(w) == 4
        w = permute(w, [2 1 3 4]); % transposed height/width to fit matconvnet
    else
        assert(ndims(w) == 2);
        w = reshape(w, [1 1 size(w)]); % expand fc layer's blob dimension to equivalent conv layer
    end
    b = caffenet.layer_vec( caffelayers(i) ).params(2).get_data();
    matconvnet.layers{ matlayers(i) }.weights{1} = w;
    matconvnet.layers{ matlayers(i) }.weights{2}(:) = b;
end

% BGR to RGB
w = matconvnet.layers{ matlayers(1) }.weights{1};
assert(size(w,3) == 3);
matconvnet.layers{ matlayers(1) }.weights{1} = w(:,:,end:-1:1,:);

save('models/alexnet_places2_convert.mat', '-struct', 'matconvnet');
