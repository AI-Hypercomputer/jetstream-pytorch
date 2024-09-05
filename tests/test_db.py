
class InMemoryDBImpl():


   def __init__(self):
       self.db = {}
   def set(self, timestamp: int, key: str, field: str, value: int) -> None:
       if key not in self.db:
           self.db[key] = {}
       self.db[key][field] = (value, -1)  


         
   def compare_and_set(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int) -> bool:
       if key not in self.db or field not in self.db[key] or expected_value != self.db[key][field][0]:
           return False
       # if self.db[key][field] != -1 and timestamp > self.db[key][field]:
       #     return False
       self.db[key][field] = (new_value, -1)
       return True   
          
  
   def compare_and_delete(self, timestamp: int, key: str, field: str, expected_value: int) -> bool:
       if key not in self.db or field not in self.db[key] or expected_value != self.db[key][field][0]:
           return False
       del  self.db[key][field]  
       return True


   def get(self, timestamp: int, key: str, field: str) -> int | None:
       if key not in self.db or field not in self.db[key]:
           return None
       return self.db[key][field][0]




   def scan(self, timestamp: int, key: str) -> list[str]:
       if key not in self.db:
           return []
       filtered_dict = { k:v[0] for k, v in self.db[key] if v[1] == -1 or timestamp <= v[1]}
       # self.db[key] = filtered_dict
       sorted_dict = sorted(self.db[key].items(), key=lambda x: x[0])
       r = [f"{k}({v})" for k, v in sorted_dict]   
       return r


   def scan_by_prefix(self, timestamp: int, key: str, prefix: str) -> list[str]:
       if key not in self.db:
           return []

       
       print(f"db----------> {self.db[key]}")   
       print(f"db----------> {self.db[key].items()}")        
       filtered_dict = { k:v1 for k, (v1, v2) in self.db[key] if v2 == -1 or timestamp <= v2}
       # self.db[key] = filtered_dict           
       sorted_dict = sorted(self.db[key].items(), key = lambda x: x[0])
       r = [f"{k}({v})" for k, v in sorted_dict if k.startswith(prefix)]   
       return r
   def set_with_ttl(self, timestamp: int, key: str, field: str, value: int, ttl: int) -> None:
       """
       Should insert the specified `value` and set its Time-To-Live
       starting at `timestamp`.
       If the `field` in the record already exists, then update its
       `value` and TTL.
       The `ttl` parameter represents the number of time units that
       this `field`-`value` pair should exist in the database,
       meaning it will be available during this interval:
       `[timestamp, timestamp + ttl)`.
       It is guaranteed that `ttl` is greater than `0`.
       """
       # default implementation
       self.set(timestamp, key, field, value)
       self.db[key][field] = (value, timestamp + ttl)


   def compare_and_set_with_ttl(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int, ttl: int) -> bool:
       """
       The same as `compare_and_set`, but should also update TTL of
       the `new_value`.
       This operation should return `True` if the field was updated
       and `False` otherwise.
       It is guaranteed that `ttl` is greater than `0`.
       """
       # default implementation
       result = self.compare_and_set(timestamp, key, field, expected_value, new_value)
       if not result:
           return result
       self.db[key][field] = (new_value, timestamp + ttl)   
       return False       


import unittest

db = InMemoryDBImpl()
db.set(0, 'employee', 'age', 20)
db.set(1, 'employee', 'article', 30)
db.set(2, 'employee', 'particle', 40)
expected = ['age(20)', 'article(30)']
db.scan_by_prefix(3, 'employee', 'a')
# expected = ['article(30)']
# db.scan_by_prefix(4, 'employee', 'ar')