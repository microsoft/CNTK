require 'example'

include Example

# Pass arguments in as one or more key-value pairs
setVitalStats("Fred",
              'age'    => 42,
              'weight' => 270
              )

# The order doesn't matter
setVitalStats("Sally",
              'age'    => 29,
              'weight' => 115,
	      'height' => 72
              )

# Can also pass a hash directly
params = {
    'ears' => 2,
    'eyes' => 2
}
setVitalStats("Bob", params)

# An empty hash is fine too
setVitalStats("Joe", {})
