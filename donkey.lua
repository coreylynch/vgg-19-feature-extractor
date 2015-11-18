require 'image'
paths.dofile('dataset.lua')

-- a cache file of the training metadata (if doesnt exist, will be created)
local testCache = paths.concat(opt.cache, 'dataCache.t7')

local loadSize   = {3, 256, 256}
local sampleSize = {3, 224, 224}

local function loadImage(path, scale)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

-- VGG preprocessing
local bgr_means = {123.68,116.779,103.939}
local function vggPreprocess(img)
  local im2 = img:clone()
  im2[{1,{},{}}] = img[{3,{},{}}]
  im2[{3,{},{}}] = img[{1,{},{}}]

  im2:mul(255)
  for i=1,3 do
    im2[i]:add(-bgr_means[i])
  end
  return im2
end

local function centerCrop(input)
   local oH = sampleSize[2]
   local oW = sampleSize[3]
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oW) -- center patch
   return out
end

-- function to load the image
local extractHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local vggPreprocessed = vggPreprocess(input)
   local out = centerCrop(vggPreprocessed)
   return out
end

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   extractLoader = torch.load(testCache)
   extractLoader.sampleHook = extractHook
else
   print('Creating test metadata')
   extractLoader = dataLoader{}
   torch.save(testCache, extractLoader)
   extractLoader.sampleHook = extractHook
end
collectgarbage()
