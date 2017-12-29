print "[lua] This is runme.lua"
-- test program for embedded lua
-- we do not need to load the library, as it was already in the interpreter
-- but let's check anyway

assert(type(example)=='table',"Don't appear to have loaded the example module. Do not run this file directly, run the embed3 executable")

print "[lua] looking to see if we have a pointer to the engine"
if type(pEngine)=="userdata" then
    print "[lua] looks good"
else
    print "[lua] nope, no signs of it"
end


-- the embedded program expects a function void onEvent(Event)
-- this is it

function onEvent(e)
    print("[Lua] onEvent with event",e.mType)
    -- let's do something with the Engine
    -- nothing clever, but ...
    if e.mType==example.Event_STARTUP then
        pEngine:start()
    elseif e.mType==example.Event_KEYPRESS then
        pEngine:accelerate(0.4)
    elseif e.mType==example.Event_MOUSEPRESS then
        pEngine:decelerate(0.4)
    elseif e.mType==example.Event_SHUTDOWN then
        pEngine:stop()
    else
        error("unknown event type")
    end
    print("[Lua] ending onEvent")
end