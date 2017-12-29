from python_threads import *

action = ActionGroup()
count = 1
for child in action.GetActionList():
    if child.val != count:
        raise RuntimeError(
            "Expected: " + str(count) + " got: " + str(child.val))
    count = count + 1

# Was seg faulting at the end here
