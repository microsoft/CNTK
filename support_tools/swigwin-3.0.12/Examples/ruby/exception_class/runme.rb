require 'example'

q = Example::IntQueue.new(10)

puts "Inserting items into intQueue"

begin
    0.upto(100) do |i|
    q.enqueue(i)
  end
rescue Example::FullError => e
  puts "Maxsize is: #{e.maxsize}"
end

puts "Removing items"

begin
  loop do
    q.dequeue()
  end
rescue Example::EmptyError => e
  ## do nothing
end

q = Example::DoubleQueue.new(1000)

puts "Inserting items into doubleQueue"

begin
  0.upto(100) do |i|
    q.enqueue(i*1.5)
  end
rescue Example::FullError => e
  puts "Maxsize is: #{e.maxsize}"
end

puts "Removing items"

begin
  loop do
    q.dequeue()
  end
rescue Example::EmptyError => e
  # do nothing
end
