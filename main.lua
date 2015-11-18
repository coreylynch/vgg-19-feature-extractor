require 'torch'
require 'paths'
require 'xlua'
require 'nn'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

if not paths.dirp(opt.cache) then
	os.execute('mkdir -p ' .. opt.cache)
end
print('Writing vectors to: ' .. opt.outFile)

paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('extract.lua')

print("\n")
print('==> Extracting VGG 19 features')
extract()
