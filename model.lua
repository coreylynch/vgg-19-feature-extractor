require 'nn'
require 'optim'
require 'loadcaffe'
require 'ccn2'
require 'util'

if not paths.dirp('model_weights') then
	print('=> Downloading VGG 19 model weights')
  os.execute('mkdir model_weights')
  local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
  local proto_url = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt'
  os.execute('wget --output-document model_weights/VGG_ILSVRC_19_layers.caffemodel ' .. caffemodel_url)
  os.execute('wget --output-document model_weights/VGG_ILSVRC_19_layers_deploy.prototxt ' .. proto_url)
end
--opt ={}
--opt.backend = "cunn"
--opt.nGPU = 1
--nGPU = opt.nGPU 
--opt.GPU = 1
--[[
   1. Load pre-trained model
   2. Remove last layer
   3. Convert model to CUDA
]]--
function getPretrainedModel(nGPU)
  local proto = 'model_weights/VGG_ILSVRC_19_layers_deploy.prototxt'
  local caffemodel = 'model_weights/VGG_ILSVRC_19_layers.caffemodel'

  if opt.backend == 'cudnn' then
      require 'cudnn'
  elseif opt.backend == 'cunn' then
      require 'cunn'
      print('using cunn backend')
  else
      error('unrecognized backend: ' .. backend)
  end

  local inputModel = loadcaffe.load(proto, caffemodel, 'ccn2')

	--[[
		Remove the Softmax, class scores, and dropout layer from the original
		network, leaving only the ReLU-ed activations immediately prior to the
		classifier.
	]]--

	for i=1,3 do
		inputModel.modules[#inputModel.modules] = nil 
	end

  --features = nn.Sequential() ; j=1; for i=40,45 do features:add(model.modules[i]) ; end
  --j=45 ; for i=1,6 do model:remove(j) ; j=j-1; end

  --final model
  model = nn.Sequential() ;
  
  features = nn.Sequential() ;
  classifier = nn.Sequential() ;
  
  features,classifier = modelParallel(inputModel)
 
  features:cuda()
  features = makeDataParallel(features,nGPU)
  model:add(features):add(classifier)  
  
  --architecture 2 addition
  model:add(nn.Linear(4096,1000))
  model:add(nn.ReLU(true))
  
  --3rd archtiecture
  model:add (nn.Linear(1000,20))

  -- L2 normalize the activations
  model:add(nn.Normalize(2))  

  model:evaluate()

  return model
end

--make the model parallelizable
--i.e features (to be parallel) and classifier (not parallel) 
function modelParallel(net)
  local classifier = nn.Sequential() ; 
  for i=40,45 do classifier:add(net.modules[i]) ; end
  j=45 ; for i=1,6 do net:remove(j) ; j=j-1; end
 return net,classifier
end

model = getPretrainedModel(nGPU)
print('=> Model')
print(model)


print('==> Converting model to CUDA')
model = model:cuda()

collectgarbage()
