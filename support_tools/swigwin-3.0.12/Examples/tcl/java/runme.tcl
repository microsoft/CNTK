catch { load ./example[info sharedlibextension] example}

JvCreateJavaVM  NULL
JvAttachCurrentThread NULL NULL
Example e1 1
Example e2 2

puts "[e1 cget -mPublicInt]"
puts "[e2 cget -mPublicInt]"

puts "[e2 Add 1 2]"
puts "[e2 Add 1.0 2.0]"
puts "[e2 Add '1' '2']"

JvDetachCurrentThread
