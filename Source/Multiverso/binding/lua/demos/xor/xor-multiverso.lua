--
-- The multiverso version train example referring from
-- https://github.com/torch/nn/blob/master/doc/training.md
--

require 'nn'

-- Load multiverso.
local multiverso = require 'multiverso'

-- Init multiverso.
multiverso.init(false)

-- Get some useful parameters from multiverso.
-- 1) The total number of workers.
multiverso.num_workers = multiverso.num_workers()
-- 2) The id for current worker.
multiverso.worker_id = multiverso.worker_id()
-- 3) Easy access to check whether this is master worker.
multiverso.is_master = multiverso.worker_id == 0

local model = nn.Sequential()
local inputs = 2
local outputs = 1
local HUs = 20
model:add(nn.Linear(inputs, HUs))
model:add(nn.Tanh())
model:add(nn.Linear(HUs, outputs))

local criterion = nn.MSECriterion()

local batchSize = 128
local batchInputs = torch.Tensor(batchSize, inputs)
local batchLabels = torch.DoubleTensor(batchSize)

for i=1,batchSize do
  local input = torch.randn(2)
  local label = 1
  if input[1]*input[2]>0 then
    label = -1;
  end
  batchInputs[i]:copy(input)
  batchLabels[i] = label
end

local params, gradParams = model:getParameters()

-- Create ArrayTableHandler for syncing parameters.
local tbh = multiverso.ArrayTableHandler:new(params:size(1), params)
-- Wait for finishing the initializing phase.
multiverso.barrier()
-- Get the initial model from the server.
params:copy(tbh:get())

for epoch=1,1000 do
  model:zeroGradParameters()
  local outputs = model:forward(batchInputs)
  local loss = criterion:forward(outputs, batchLabels)
  local dloss_doutput = criterion:backward(outputs, batchLabels)
  model:backward(batchInputs, dloss_doutput)

  -- Sync parameters:
  -- 1) Add the gradients (delta value) to the server.
  tbh:add(-0.01 * gradParams)
  -- 2) (Optional) Sync all workers after each epoch.
  multiverso.barrier()
  -- 3) Fetch the newest value from the server.
  params:copy(tbh:get())

  -- Print should also only exist in master worker.
  if multiverso.is_master then
    print(epoch)
  end
end

-- Only test in master worker.
if multiverso.is_master then
  local x = torch.Tensor({
    {0.5, 0.5},
    {0.5, -0.5},
    {-0.5, 0.5},
    {-0.5, -0.5}
  })
  print(model:forward(x))
end

-- Remember to shutdown at last.
multiverso.shutdown()
