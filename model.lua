require 'nn'
require 'optim'
require 'loadcaffe'
require 'util'
require 'cunn'


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
function getPretrainedModel(nGPU)
  local proto = 'model_weights/VGG_ILSVRC_19_layers_deploy.prototxt'
  local caffemodel = 'model_weights/VGG_ILSVRC_19_layers.caffemodel'

  if opt.backend == 'cudnn' then
      require 'cudnn'
  elseif opt.backend == 'cunn' then
      require 'cunn'
      print('using cunn backend')
  elseif opt.backend == 'ccn2' then 
      require 'ccn2'
      print('using ccn2')
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


  --final model
  model = nn.Sequential() ;
  
  features = nn.Sequential() ;
  classifier = nn.Sequential() ;
   
  --return modules for them to be parallelized in the next step   
  features,classifier = modelParallel(inputModel)
  features:cuda()

  --multi gpu use (caution no dependencies) 
  features = makeDataParallel(features,nGPU)
  model:add(features):add(classifier)  
  
  model:add(nn.Linear(4096,1000))
  model:add(nn.ReLU(true))
  model:add (nn.Linear(1000,20))

  -- L2 normalize the activations
  model:add(nn.Normalize(2))  

  model:evaluate()

  return model
end

--make the model parallelizable
--i.e net (to be parallel) and classifier (non parallel) 
function modelParallel(net)
  --classifier: is the latter half which is non parallelized , add the parallelizable modules here
  local classifier = nn.Sequential() ; 
  for i=40,45 do classifier:add(net.modules[i]) ; end

  --net: first half which is parallelized , remove the classifier modules 
  j=45 ; for i=1,6 do net:remove(j) ; j=j-1; end
 return net,classifier
end

--run the model in multi gpu's if allowed
model = getPretrainedModel(opt.nGPU)
print('=> Model')
print(model)


print('==> Converting model to CUDA')
model = model:cuda()

collectgarbage()
