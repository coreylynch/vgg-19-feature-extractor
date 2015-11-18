local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at worker.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   if opt.nThreads > 0 then
      local options = opt -- make an upvalue to serialize over to worker threads
      donkeys = Threads(
         opt.nThreads,
         function()
            require 'torch'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting worker with id: %d seed: %d', tid, seed))
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

donkeys:synchronize()

nExtract = 0
donkeys:addjob(function() return extractLoader:size() end, function(c) nExtract = c end)
donkeys:synchronize()
assert(nExtract > 0, "Failed to get nExtract")
print('Number of images to embed: ', nExtract)