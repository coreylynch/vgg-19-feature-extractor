require 'xlua'
require 'json'

file = io.open(opt.outFile,'w')

function extract()
   cutorch.synchronize()

   for i=1, nExtract/opt.batchSize do -- nExtract is set in data.lua
      collectgarbage()
      xlua.progress(i, nExtract/opt.batchSize)
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, ids = extractLoader:get(indexStart, indexEnd)
            return inputs, ids
         end,
         -- callback that is run in the main thread once the work is done
         extractBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

end -- of extract()

local inputs = torch.CudaTensor()

function jsonStringFromCudaTensor(c)
  t = {}
  for i = 1,c:size(1) do
    t[i] = c[i]
  end
  return json.encode(t)
end

function extractBatch(inputsCPU, idsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)

   local outputs = model:forward(inputs)
   collectgarbage()

   for i = 1,outputs:size(1) do
      local jsonString = jsonStringFromCudaTensor(outputs[i])
      local id = idsCPU[i]
      file:write(id.."\t"..jsonString.."\n")
      file:flush()
   end

   collectgarbage()
   cutorch.synchronize()
end
