from utils import of_arity

class MathematicalSet:
    def __new__(cls, set_cls, *predicates):
        if not isinstance(set_cls, type):
            raise TypeError()
        if len(predicates) == 1 and isinstance(predicates[0], list):
            predicates = predicates[0]
        if not all(of_arity(pred, 1) for pred in predicates):  # make sure pred type is valid
            raise ValueError()
        
        return super().__new__(cls)

    def __init__(self, cls, *predicates):
        if len(predicates) == 1 and isinstance(predicates[0], list):
            predicates = predicates[0]

        self._cls = cls
        self._predicates = remove_duplicates(predicates)

    @property
    def cls(self):
        return self._cls
    
    @property
    def predicates(self):
        return self._predicates
    
    def __repr__(self):
        return f'Set({self.cls.__name__}, {[pred.__name__ for pred in self.predicates]})'
    
    def __eq__(self, set2):
        # Order of the predicates matters
        return vars(self) == vars(set2)
    
    def __contains__(self, obj):
        if not isinstance(obj, self.cls):
            return False
        return all(pred(obj) for pred in self.predicates)
    
    def __neg__(self):
        return self.complement()
    
    def __pos__(self):
        return self
    
    def __and__(self, set2):
        return self.intersection(set2)
    
    def __or__(self, set2):
        return self.union(set2)
    
    def __sub__(self, set2):
        return self.difference(set2)

    def complement(self):
        complement_pred = lambda obj: not all(pred(obj) for pred in self.predicates)
        complement_pred.__name__ = 'complement_predicate'
        return Set(self.cls, complement_pred)
    
    def intersection(self, set2):
        self._check_type(set2)
        return Set(self.cls, self.predicates + set2.predicates)

    def union(self, set2):
        self._check_type(set2)
        union_pred = lambda obj: (all(pred(obj) for pred in self.predicates) or 
                                  all(pred(obj) for pred in set2.predicates))
        union_pred.__name__ = 'union_predicate'
        return Set(self.cls, union_pred)

    def difference(self, set2):
        return self.intersection(set2.complement())
    
    def is_subset(self, set2):
        '''
        Returns True if all the predicates in set2 are in self, otherwise False.
        '''
        self._check_type(set2)
        return all(pred in self.predicates for pred in set2.predicates)
    
    def add_predicates(self, *predicates):
        if len(predicates) == 1 and isinstance(predicates[0], list):
            predicates = predicates[0]
        return Set(self.cls, *self.predicates, *predicates)

    def _check_type(self, set2):
        if not isinstance(set2, Set):
            raise TypeError(f'Expected a MathematicalSet, got {type(set2).__name__} instead.')
        if self.cls is not set2.cls:
            raise ValueError('The cls attribute of both sets must be the same.')


def remove_duplicates(seq):
    '''
    Returns a list containing the items in seq in order with duplicates removed.
    '''
    elems = set()
    return [x for x in seq if not (x in elems or elems.add(x))]

def negate(pred):
    '''
    Returns a function that negates the output of pred.
    '''
    negation = lambda obj: not pred(obj)
    negation.__name__ = f'not_{pred.__name__}'
    return negation

# Set alias
Set = MathematicalSet