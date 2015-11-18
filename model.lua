require 'nn'
require 'optim'
require 'loadcaffe'

if not paths.dirp('model_weights') then
	print('=> Downloading VGG 19 model weights')
  os.execute('mkdir model_weights')
  local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
  local proto_url = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt'
  os.execute('wget --output-document model_weights/VGG_ILSVRC_19_layers.caffemodel ' .. caffemodel_url)
  os.execute('wget --output-document model_weights/VGG_ILSVRC_19_layers_deploy.prototxt ' .. proto_url)
end

--[[
   1. Load pre-trained model
   2. Remove last layer
   3. Convert model to CUDA
]]--
function getPretrainedModel()
  local proto = 'model_weights/VGG_ILSVRC_19_layers_deploy.prototxt'
  local caffemodel = 'model_weights/VGG_ILSVRC_19_layers.caffemodel'

  if opt.backend == 'cudnn' then
      require 'cudnn'
  elseif backend == 'cunn' then
      print('using cunn backend')
  else
      error('unrecognized backend: ' .. backend)
  end

  local model = loadcaffe.load(proto, caffemodel, opt.backend)

	--[[
		Remove the Softmax, class scores, and dropout layer from the original
		network, leaving only the ReLU-ed activations immediately prior to the
		classifier.
	]]--
	for i=1,3 do
		model.modules[#model.modules] = nil 
	end

	-- L2 normalize the activations
  model:add(nn.Normalize(2))

  model:evaluate()

  return model
end

model = getPretrainedModel()
print('=> Model')
print(model)


print('==> Converting model to CUDA')
model = model:cuda()

collectgarbage()