require 'torch'
local ffi = require 'ffi'
local util = require 'multiverso.util'

local mv = {}

ffi.cdef[[
    typedef void* TableHandler;
    void MV_Init(int* argc, char* argv[]);
    void MV_ShutDown();
    void MV_Barrier();
    int MV_NumWorkers();
    int MV_WorkerId();
    int MV_ServerId();
]]

package.cpath = '/usr/local/lib/?.so;' .. package.cpath
libmv_path = package.searchpath('libmultiverso', package.cpath, '')
if libmv_path == nil then
    print([[
[Error] Multiverso shared object, `libmultiverso.so`, NOT FOUND!
Please build & install `multiverso` according to the instruction [1].
[1] https://github.com/Microsoft/multiverso#build]])
    return
end
libmv = ffi.load(libmv_path, 'true')

mv.ArrayTableHandler = require('multiverso.ArrayTableHandler')
mv.MatrixTableHandler = require('multiverso.MatrixTableHandler')

function mv.init(sync)
    sync = sync or false  -- false for the default value of sync
    -- the first argument will be ignored. So we put a placeholder here
    args = {""}
    if sync then
        table.insert(args, "-sync=true")
    end
    argc = ffi.new("int[1]", #args)
    argv = ffi.new("char*[?]", #args)
    for i = 1, #args do
        argv[i - 1] = ffi.new("char[1]")
        ffi.copy(argv[i - 1], args[i])
    end
    libmv.MV_Init(argc, argv)
end

function mv.barrier()
    libmv.MV_Barrier()
end

function mv.shutdown()
    libmv.MV_ShutDown()
end

function mv.num_workers()
    return libmv.MV_NumWorkers()
end

function mv.worker_id()
    return libmv.MV_WorkerId()
end

function mv.server_id()
    return libmv.MV_ServerId()
end

return mv
