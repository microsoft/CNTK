require 'example'

# create a zoo
zoo = Example::Zoo.new

begin
  # Add in an couple of animals
  tiger1 = Example::Animal.new("tiger1")
  zoo.add_animal(tiger1)
  
  # unset variables to force gc
  tiger1 = nil
end

GC.start

#  Now get the tiger again
tiger2 = zoo.get_animal(0)

# Call a method to verify the animal is still valid and not gc'ed
if tiger2.get_name != "tiger1"
    raise RuntimeError, "Wrong animal name"
end
