require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('dataLoader')

function dataset:__init(...)

   local imagePathFile = opt.data;

   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- find the image path names
   self.imagePaths = torch.CharTensor()  -- path to each image in dataset
   self.ids = torch.CharTensor()  -- path to each image in dataset

   --==========================================================================
   print('loading the large list of image paths to self.imagePath and caching')

   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. imagePathFile .. "' |"
                                           .. cut .. " -f1 -d' '"))

   local idMaxLength = tonumber(sys.fexecute(cut .. " " .. imagePathFile ..
                                               " -f1 | " .. wc .. " -L | " .. cut 
                                               .. " -f1") + 1) + 1
   local pathMaxLength = tonumber(sys.fexecute(cut .. " " .. imagePathFile ..
                                               " -f2 | " .. wc .. " -L | " .. cut 
                                               .. " -f1") + 1) + 1
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(idMaxLength > 0, "ids aren't missing?")
   assert(pathMaxLength > 0, "paths of files are length 0?")

   self.imagePaths:resize(length, pathMaxLength):fill(0)
   self.ids:resize(length, idMaxLength):fill(0)
   local path_data = self.imagePaths:data()
   local id_data = self.ids:data()
   local count = 0
   for line in io.lines(imagePathFile) do
      local id_path_label = line:split("\t")
      ffi.copy(id_data, id_path_label[1])
      ffi.copy(path_data, id_path_label[2])
      
      id_data = id_data + idMaxLength
      path_data = path_data + pathMaxLength
      
      if self.verbose and count % 10000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end

   self.numSamples = self.imagePaths:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
end

-- size()
function dataset:size(class, list)
  return self.numSamples
end

-- converts a table of samples to a clean tensor
local function tableToOutput(self, dataTable)
   local data
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 3)
   data = torch.Tensor(quantity,
           3, 224, 224)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
   end
   return data
end

-- Gets a range of images from the test set.
function dataset:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local idsTable = {}
   for i=1,quantity do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePaths[indices[i]]))
      local id = ffi.string(torch.data(self.ids[indices[i]]))
      local out = self:sampleHook(imgpath)
      table.insert(dataTable, out)
      table.insert(idsTable, id)
   end
   local data = tableToOutput(self, dataTable)
   return data, idsTable
end

return dataset
