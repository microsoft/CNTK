require 'example'

begin
  begin
    # Create an animal and zoo
    tiger1 = Example::Animal.new("tiger1")
    zoo = Example::Zoo.new
  
    # At the animal to the zoo - this will transfer ownership
    # of the underlying C++ object to the C++ zoo object
    zoo.add_animal(tiger1)

    # get the id of the tiger
    id1 = tiger1.object_id

    # Unset the tiger
    tiger1 = nil
  end

  # Force a gc
  GC.start

  # Get the tiger and its id
  tiger2 = zoo.get_animal(0)
  id2 = tiger2.object_id

  # The ids should not be the same
  if id1==id2
# Not working - needs checking/fixing
#    raise RuntimeError, "Id's should not be the same"
  end

  zoo = nil
end

GC.start

# This method is no longer valid since the zoo freed the underlying
# C++ object
ok = false
begin
  puts tiger2.get_name
rescue ObjectPreviouslyDeleted => error
  ok = true
end

raise(RuntimeError, "Incorrect exception raised - should be ObjectPreviouslyDeleted") unless ok
