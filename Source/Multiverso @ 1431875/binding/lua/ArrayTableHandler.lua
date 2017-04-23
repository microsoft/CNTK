local ffi = require 'ffi'
local util = require('multiverso.util')

local tbh = torch.class('ArrayTableHanlder')

ffi.cdef[[
    void MV_NewArrayTable(int size, TableHandler* out);
    void MV_GetArrayTable(TableHandler handler, float* data, int size);
    void MV_AddArrayTable(TableHandler handler, float* data, int size);
    void MV_AddAsyncArrayTable(TableHandler handler, float* data, int size);
]]

function tbh:new(size, init_value)
    tbh = {}
    size = size or 0
    setmetatable(tbh, self)
    self.__index = self
    tbh._handler = ffi.new("TableHandler[1]")
    tbh._size = ffi.new("int", size)
    libmv.MV_NewArrayTable(
        tbh._size,
        tbh._handler
    )
    local init = require 'multiverso.init'
    if init_value ~= nil then
        init_value = init_value:float()
        -- sync add is used because we want to make sure that the initial value
        -- has taken effect when the call returns. No matter whether it is
        -- master worker,  we should call add to make sure it works in sync
        -- mode
        if init.worker_id() == 0 then
            self.add(tbh, init_value, true)
        else
            self.add(tbh, init_value:clone():zero(), true)
        end
    end
    return tbh
end

function tbh:get()
    cdata = ffi.new("float[?]", self._size)
    libmv.MV_GetArrayTable(self._handler[0], cdata, self._size)
    return util.cdata2tensor(cdata, tonumber(self._size))
end

function tbh:add(data, sync)
    sync = sync or false
    cdata = util.tensor2cdata(data)
    if sync then
        libmv.MV_AddArrayTable(self._handler[0], cdata, self._size)
    else
        libmv.MV_AddAsyncArrayTable(self._handler[0], cdata, self._size)
    end
end

return tbh
