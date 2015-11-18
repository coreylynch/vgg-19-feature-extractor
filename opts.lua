local M = { }

function M.parse(arg)
    local defaultDir = 'vgg_features/'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Extract VGG 19 Features')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-data',
               'path_to_images',
               'File of tab separated "image_id path_to_image_on_disk"')
    cmd:option('-cache',
               'cache',
               'caching image metadata')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | cunn')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-outFile',     'vecs', 'File to write (image_id \t json encoded vgg feature vector)')
    ------------- Data options ------------------------
    cmd:option('-nThreads',        4, 'number of data loading threads to initialize')
    cmd:option('-batchSize',       128,   'mini-batch size')
    cmd:text()

    local opt = cmd:parse(arg or {})
    return opt
end

return M