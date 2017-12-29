#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'li_factory'

circle = Li_factory::Geometry.create(Li_factory::Geometry::CIRCLE)
r = circle.radius()
if (r != 1.5)
	raise RuntimeError, "Invalid value for r"
end

new_circle = circle.clone()
r = new_circle.radius()
if (r != 1.5)
	raise RuntimeError, "Invalid value for r"
end

point = Li_factory::Geometry.create(Li_factory::Geometry::POINT)
w = point.width()

if (w != 1.0)
	raise RuntimeError, "Invalid value for w"
end

new_point = point.clone()
w = new_point.width()

if (w != 1.0)
	raise RuntimeError, "Invalid value for w"
end
