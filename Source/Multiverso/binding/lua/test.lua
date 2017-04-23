#!/usr/bin/env lua

require 'torch'

mv = require('multiverso')

local mv_test = torch.TestSuite()
local mv_tester = torch.Tester()

function Set(list)
  local set = {}
  for _, l in ipairs(list) do set[l] = true end
  return set
end

function mv_test.testArray()
    size = 100000
    tbh = mv.ArrayTableHandler:new(size)
    mv.barrier()

    for i = 1, 1000 do
        print(tbh:get()[{{1, 10}}])
        tbh:add(torch.range(1, size))
        tbh:add(torch.range(1, size))
        mv.barrier()
    end
end

function mv_test.testMatrix()
    num_row = 11
    num_col = 10
    size = num_row * num_col
    num_workers = mv.num_workers()
    tbh = mv.MatrixTableHandler:new(num_row, num_col)
    mv.barrier()

    for i = 1, 20 do
        row_ids = {0, 1, 5, 10}
        row_ids_set = Set(row_ids)
        tbh:add(torch.range(1, size))
        data = torch.range(
            row_ids[1] * num_col + 1,
            row_ids[1] * num_col + num_col
        )
        for j = 2, #row_ids do
            row_id = row_ids[j]
            data = torch.cat(data, torch.range(
                row_id * num_col + 1,
                row_id * num_col + num_col
            ))
        end
        tbh:add(data, row_ids)
        mv.barrier()
        data = tbh:get()
        mv.barrier()
        for j = 1, data:size(1) do
            for k = 1, data:size(2) do
                expected = ((j - 1) * num_col + k) * i * num_workers
                if row_ids_set[j - 1] then
                    expected = expected + ((j - 1) * num_col + k) * i * num_workers
                end
                mv_tester:eq(expected, data[j][k])
            end
        end
        data = tbh:get(row_ids)
        mv.barrier()
        for j = 1, data:size(1) do
            for k = 1, data:size(2) do
                expected = (row_ids[j] * num_col + k) * i * num_workers * 2
                mv_tester:eq(expected, data[j][k])
            end
        end
    end
end

mv.init()
mv_tester:add(mv_test)
mv_tester:run()
mv.shutdown()
