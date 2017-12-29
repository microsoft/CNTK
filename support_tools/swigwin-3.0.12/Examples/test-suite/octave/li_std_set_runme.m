li_std_set

s = set_string()

s.append("a")
s.append("b")
s.append("c")

sum = ""
for i in s:
    sum = sum + i

if (sum != "abc")
    error

i = s.__iter__()
if i.next() != "a":
    error
if i.next() != "b":
    error
if i.next() != "c":
    error


b = s.begin()
e = s.end()
sum = ""
while (b != e):    
    sum = sum + b.next()
if sum != "abc":
    error

b = s.rbegin()
e = s.rend()
sum = ""
while (b != e):    
    sum = sum  + b.next()

if sum != "cba":
    error



si = set_int()

si.append(1)
si.append(2)
si.append(3)
i = si.__iter__()

if i.next() != 1:
    error
if i.next() != 2:
    error
if i.next() != 3:
    error




i = s.begin()
i.next()
s.erase(i)

b = s.begin()
e = s.end()
sum = ""
while (b != e):    
    sum = sum + b.next()
if sum != "ac":
    error


b = s.begin()
e = s.end()
if e - b != 2:
    error
    
m = b + 1
if m.value() != "c":
    error



s = pyset()
s.insert((1,2))
s.insert(1)
s.insert("hello")


sum = ()
for i in s:
    sum = sum  + (i,)

if sum != (1, 'hello', (1, 2)):
    error
