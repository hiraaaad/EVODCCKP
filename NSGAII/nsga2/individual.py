# from NSGAII_copy.nsga2.utils import np
from nsga2.utils import np

class Individual(object):
    def __init__(self, length):
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = np.zeros((1,length))
        self.objectives = [1e12,1e1]

    def __deepcopy__(self, memodict={}):
        copy_object = Individual(length=len(self.features))
        copy_object.rank = self.rank
        copy_object.crowding_distance = self.crowding_distance
        copy_object.domination_count = self.domination_count
        copy_object.dominated_solutions = self.dominated_solutions
        copy_object.features = self.features.copy()
        copy_object.objectives = self.objectives.copy()
        return copy_object

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)
