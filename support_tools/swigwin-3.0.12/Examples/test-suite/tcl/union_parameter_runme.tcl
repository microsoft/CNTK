if [ catch { load ./union_parameter[info sharedlibextension] union_parameter} err_msg ] {
        puts stderr "Could not load shared object:\n$err_msg"
}

set event [SDL_Event]

for { set i 0 } { $i < 2 } { incr i } {
#    puts -nonewline "Loop $i: "
    set evAvailable [SDL_PollEvent $event]
    set evType [$event cget -type]
#    puts "evType = $evType"

    if { $evType == 1 } {
        set specEvent [$event cget -active]
#        puts "specEvent = $specEvent"
        set type [$specEvent cget -type]
        if { $type != $evType } {
            error "Type $type should be $evType"
        }
        set gain   [$specEvent cget -gain]
        set state  [$specEvent cget -state]
#        puts "gain=$gain state=$state"
    }
    if { $evType == 2 } {
        set specEvent [$event cget -key]
#        puts "specEvent = $specEvent"
        set type [$specEvent cget -type]
        if { $type != $evType } {
            error "Type $type should be $evType"
        }
        set which  [$specEvent cget -which]
        set state  [$specEvent cget -state]
#        puts "which=$which state=$state"
    }
#    puts ""
}
