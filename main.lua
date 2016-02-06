#!/usr/bin/env luajit

require 'torch'
require 'nn'
require 'image'
require 'libcv'
require 'libutil'
require 'optim'
local threads = require 'threads'
local stringx = require 'pl.stringx'
local jdb = require 'jdb'

cmd = torch.CmdLine()
cmd:option('-seed', 1337)
cmd:option('-num_threads', 4)
cmd:option('-db_dir', 'data/db')

cmd:option('-net', 'alexnetowt')
cmd:option('-batch_size', 32)
cmd:option('-lr', 0.001)
cmd:option('-mom', 0.9)
cmd:option('-wd', 0.0001)

cmd:option('-alphastd', 0.5)
local opt = cmd:parse(arg)
io.stdout:setvbuf('line')

-- print cmd arguments
local net_fname = opt.net
for i = 1,#arg do
   io.write(arg[i] .. ' ')
      net_fname = net_fname .. '_' .. arg[i]
   end
print()

threads.serialization('threads.sharedserialize')
local pool = threads.Threads(opt.num_threads, function()
   require 'image'
   require 'libcv'
   torch.manualSeed(opt.seed)
end)

-- create db of unmodified images
if false then
   local max_files = 2000000
   local class_idx = 1
   local class_map = {}
   local classes = torch.FloatTensor(max_files)
   local db = jdb.Write(opt.db_dir .. '/db', max_files)
   local i = 1

   print('read train')
   os.execute('find data/data/train/ -type f | sort -R --random-source=/dev/zero > data/train.txt')
   -- md5sum data/train.txt -> a79fe5542d3548cf1c7b8abe3e6d8349
   for line in io.lines('data/train.txt') do
      db:append(io.open(line):read('*a'))
      local path = stringx.split(line, '/')
      local cls = path[#path - 1]
      if class_map[cls] == nil then
         class_map[cls] = class_idx
         class_idx = class_idx + 1
      end
      classes[i] = class_map[cls]
      i = i + 1
   end
   local num_train = i - 1

   print('read val')
   os.execute('find data/data/val/ -type f | sort -R --random-source=/dev/zero > data/val.txt')
   -- md5sum data/val.txt -> 59316afa43af3546afcaa79d07473a7b
   for line in io.lines('data/val.txt') do
      db:append(io.open(line):read('*a'))
      local path = stringx.split(line, '/')
      local cls = path[#path - 1]
      classes[i] = class_map[cls]
      i = i + 1
   end
   db:close()
   classes = classes[{{1,i - 1}}]:clone()
   torch.save(opt.db_dir .. '/data.t7', {classes, class_map, num_train})
   os.exit()
else
   db = jdb.Read(opt.db_dir .. '/db')
   assert(db:size() == 1331167)
   classes, class_map, num_train = table.unpack(torch.load(opt.db_dir .. '/data.t7'))
end

-- verify that all images are valid
if false then
   t = torch.tic()
   for i = 1,db:size() do
      local img = db:get(i)
      pool:addjob(function()
            image.decompress(img, 3, 'byte')
         end)
      if i % 1024 == 0 then
         print(i)
      end
   end
   pool:synchronize()
   print(torch.toc(t))
   os.exit()
end

local function mul32(a,b)
   return {a[1]*b[1]+a[2]*b[4], a[1]*b[2]+a[2]*b[5], a[1]*b[3]+a[2]*b[6]+a[3],
           a[4]*b[1]+a[5]*b[4], a[4]*b[2]+a[5]*b[5], a[4]*b[3]+a[5]*b[6]+a[6]}
end

local mean, eig_vec, eig_val

local function make_patch(img, dst, test)
   local src = image.decompress(img, 3, 'float')
   local height = src:size(2)
   local width = src:size(3)

   local m = {1, 0, -width / 2, 0, 1, -height / 2}

   -- scale shorter side to 256
   local scale = math.max(256 / height, 256 / width)
   m = mul32({scale, 0, 0, 0, scale, 0}, m)
   height = height * scale
   width = width * scale

   -- translate
   if not test then
      local dw = (width - 224) / 2
      local dh = (height - 224) / 2
      local trans_x = torch.uniform(-dw, dw)
      local trans_y = torch.uniform(-dh, dh)
      m = mul32({1, 0, -trans_x, 0, 1, -trans_y}, m)
   end

   -- horizontal flip
   if not test and torch.uniform() < 0.5 then
      m = mul32({-1, 0, 0, 0, 1, 0}, m)
   end

   m = mul32({1, 0, 224 / 2, 0, 1, 224 / 2}, m)
   cv.warp_affine(src, dst, torch.FloatTensor(m))

   -- lighting augmentation
   if not test and opt.alphastd > 0 then
      local alpha = torch.FloatTensor(3):normal(0, opt.alphastd)
      local color_diff = torch.mv(eig_vec, alpha:cmul(eig_val))
      dst:add(color_diff:view(3,1,1):expandAs(dst))
   end
end

-- compute image statistics
if false then
   torch.manualSeed(opt.seed)
   local N = 10000
   local pixels = torch.FloatTensor(N, 3, 224, 224):zero()
   for i = 1,N do
      make_patch(db:get(i), pixels[i])
   end
   pixels = pixels:transpose(1, 2):reshape(3, N * 224 * 224)
   local mean = pixels:mean(2)
   pixels:add(-1, mean:expandAs(pixels))

   local cov = torch.mm(pixels, pixels:t()):double() / pixels:size(2)
   local eig_vec, eig_val = torch.svd(cov)

   print(mean)
   print(eig_vec)
   print(eig_val)

   torch.save(opt.db_dir .. '/stats.t7', {mean:float(), eig_vec:float(), eig_val:float()})
   os.exit()
else
   mean, eig_vec, eig_val = table.unpack(torch.load(opt.db_dir .. '/stats.t7'))
   mean:view(3, 1, 1):expand(3, 224, 224)
end

torch.manualSeed(opt.seed)
if opt.net == 'alexnetowt' then
   net = nn.Sequential()
   net:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialMaxPooling(3,3,2,2))
   net:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialMaxPooling(3,3,2,2))
   net:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialMaxPooling(3,3,2,2))
   net:add(nn.View(256*6*6))
   net:add(nn.Dropout(0.5))
   net:add(nn.Linear(256*6*6, 4096))
   net:add(nn.ReLU(true))
   net:add(nn.Dropout(0.5))
   net:add(nn.Linear(4096, 4096))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(4096, 1000))
   net:add(nn.LogSoftMax())
elseif opt.net == 'vgg' then
   local modelType = 'A'
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end
   net = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            net:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            net:add(conv3)
            net:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end
   net:add(nn.View(512*7*7))
   net:add(nn.Linear(512*7*7, 4096))
   net:add(nn.Threshold(0, 1e-6))
   net:add(nn.Dropout(0.5))
   net:add(nn.Linear(4096, 4096))
   net:add(nn.Threshold(0, 1e-6))
   net:add(nn.Dropout(0.5))
   net:add(nn.Linear(4096, 1000))
   net:add(nn.LogSoftMax())
end

require 'cutorch'
require 'cunn'
require 'cudnn'

net = net:cuda()
criterion = nn.ClassNLLCriterion():cuda()

cudnn.benchmark = true
cudnn.fastest = true
cudnn.convert(net, cudnn)

local params, grads = net:getParameters()
local x_batch = torch.CudaTensor(opt.batch_size, 3, 224, 224)
local y_batch = torch.CudaTensor(opt.batch_size)
local x_batch_ = torch.FloatTensor(opt.batch_size, 3, 224, 224)
local pred_val = torch.CudaTensor()
local pred_ind = torch.CudaTensor()
local train_err = 6.9
local val_errs = {}
local lr_reductions = 0
local optim_state = {
	learningRate = opt.lr,
	learningRateDecay = 0.0,
	momentum = opt.mom,
	nesterov = true,
	dampening = 0.0,
	weightDecay = opt.wd,
}
local function feval()
	return criterion.output, grads
end

-- first run so that the cudnn auto-tunner does not affect the time measurements
net:forward(x_batch)
criterion:forward(net.output, y_batch:fill(1))
criterion:backward(net.output, y_batch)
net:backward(x_batch, criterion.gradInput)

torch.manualSeed(opt.seed)
cutorch.synchronize()
start = torch.tic()
for epoch=1,100 do
   -- train
   net:training()
   collectgarbage()
   for t = 1,num_train - opt.batch_size,opt.batch_size do
      for i = 1,opt.batch_size do
         local img = db:get(t + i - 1)
         pool:addjob(function()
               make_patch(img, x_batch_[i])
               x_batch_[i]:add(-1, mean)
            end)
      end
      pool:synchronize()
      x_batch:copy(x_batch_)
      y_batch:copy(classes[{{t, t + opt.batch_size - 1}}])

      net:forward(x_batch)
      train_err = 0.99 * train_err + 0.01 * criterion:forward(net.output, y_batch)
      grads:zero()
      criterion:backward(net.output, y_batch)
      net:backward(x_batch, criterion.gradInput)

      optim.sgd(feval, params, optim_state)

      if t % (500 * opt.batch_size) == 1 then
         print(('train epoch=%d t=%d train_nll=%.2f lr=%.1e time(h)=%.1f'):format(epoch, t, train_err, opt.lr, torch.toc(start) / 3600))
      end
   end

   -- save
   net:apply(function(module) 
         module.output = torch.CudaTensor()
         module.gradInput = torch.CudaTensor()
      end)
   torch.save(('net/%s_%d.t7'):format(net_fname, epoch), {net, optim_state})

   -- validate
   local val_err = 0
   local val_err_cnt = 0
   net:evaluate()
   collectgarbage()
   for t = num_train,db:size() - opt.batch_size,opt.batch_size do
      for i = 1,opt.batch_size do
         local img = db:get(t + i - 1)
         pool:addjob(function()
               make_patch(img, x_batch_[i], true)
               x_batch_[i]:add(-1, mean)
            end)
      end
      pool:synchronize()
      x_batch:copy(x_batch_)
      y_batch:copy(classes[{{t, t + opt.batch_size - 1}}])

      net:forward(x_batch)
      torch.max(pred_val, pred_ind, net.output, 2)
      val_err = val_err + pred_ind:ne(y_batch):sum()
      val_err_cnt = val_err_cnt + opt.batch_size
   end
   print(('val epoch=%d val_top1=%.4f'):format(epoch, val_err / val_err_cnt))
   table.insert(val_errs, val_err / val_err_cnt)

   -- reduce learning rate
   if #val_errs >= 5 and val_errs[#val_errs] + 0.01 > val_errs[#val_errs - 4] then
      if lr_reductions == 3 then
         break
      end
      opt.lr = opt.lr / 10
      lr_reductions = lr_reductions + 1
   end
end
cutorch.synchronize()
print(torch.toc(start))
