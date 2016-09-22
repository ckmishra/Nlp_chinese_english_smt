from pygments.token import Other
try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

class Skill(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        print 'New Level:', description
        return
    def __cmp__(self, other):
        return cmp(other.priority,self.priority)
    
    def __eq__(self,other):
        return self.priority == other.priority and self.description == other.description
    
def is_in_queue(x, q):
   with q.mutex:
      return x in q.queue
q = Q.PriorityQueue()

q.put(Skill(5, 'Proficient'))
q.put(Skill(10, 'Expert'))
q.put(Skill(1, 'Novice'))

skill = Skill(1,"Test")
if(not(is_in_queue(skill,q))):
    q.put(skill);

while not q.empty():
    next_level = q.get()
    print 'Processing level:', next_level.description