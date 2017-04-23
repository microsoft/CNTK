--
-- The train example referring from
-- https://github.com/torch/nn/blob/master/doc/training.md
--

require 'nn'

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

for epoch=1,2000 do
  model:zeroGradParameters()
  local outputs = model:forward(batchInputs)
  local loss = criterion:forward(outputs, batchLabels)
  local dloss_doutput = criterion:backward(outputs, batchLabels)
  model:backward(batchInputs, dloss_doutput)
  model:updateParameters(0.01)
end

local x = torch.Tensor({
  {0.5, 0.5},
  {0.5, -0.5},
  {-0.5, 0.5},
  {-0.5, -0.5}
})
print(model:forward(x))
