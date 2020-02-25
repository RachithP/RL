import numpy as np
from .SUMTree import *

class PriorityBuffer:
    """
        STOCHASTIC PRIORITIZATION - Dividing into batches and picking random element(priority) from each batch
    """
    def __init__(self, max_size):
        '''
        - Remember that our tree is composed of a sum tree that contains the priority scores at his leaf and also a data array
        - We don't use deque because it means that at each timestep our experiences change index by one.
        - We prefer to use a simple array and to overwrite when the memory is full.
        '''
        self.epsilon = 0.01                
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.abs_upper_error_bound = 1
        self.tree = SUMTree(max_size)

    def store(self, experience):
        '''
        - Store a new experience in our tree
        - Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DQN model)
        '''
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])             # index of leaf nodes in the tree starts from end till self.tree.capacity+1

        if max_priority == 0:                                                   # This happens when no priority element is added before and this insertion would be the first.
            max_priority = self.abs_upper_error_bound                           # Assign max priority

        self.tree.insert(max_priority, experience)

    def sample(self, batch_size):
        '''
        - First, to sample a minibatch of k size, the range [0, priority_total(i.e. self.tree.total)] is / into k ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element
        '''
        experiences = []
        IS_weights = np.empty((batch_size, 1), dtype=np.float64)
        leaf_indices = np.empty((batch_size,), dtype=np.int64)

        priority_segment = self.tree.total() / batch_size                   # divide total priority into segments of equal length for uniform sampling

        self.beta = np.min([1.0, self.beta+self.beta_increment])            # max value beta can take is 1 i.e. towards the end of training

        probability_min_priority = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()

        max_weight = (batch_size*probability_min_priority)**(-self.beta)

        for i in range(batch_size):                                     # sample values from each batch uniformly
            
            a, b = priority_segment*i, priority_segment*(i+1)           # indices from lower and upper bound

            rand_priority = np.random.uniform(a, b)                     # sample uniformly from each segment

            data, priority, leaf_index = self.tree.get_leaf(rand_priority)  # retrieve experience, priortity and priority index in tree 

            probability_priority = priority / self.tree.total()         # probability of each priority value
    
            IS_weights[i, 0] =  (batch_size*probability_priority)**(-self.beta) / max_weight        # higher probability -> higher priority -> higher error -> more weight

            leaf_indices[i] = leaf_index

            experiences.append(data)

        return experiences, IS_weights, leaf_indices

    
    def update_priorities_batch(self, tree_indices, abs_td_errors):
        '''
            Update priorities on the tree at tree_index with abs_error. 
            Implementing Proportional prioritization!
        '''
        abs_td_errors += self.epsilon                                                   # add a const to prevent unstability
        clipped_errors = np.minimum(abs_td_errors, self.abs_upper_error_bound)          # clip with the upper limit
        clipped_errors_alpha = np.float_power(clipped_errors, self.alpha)               # scale these clipped errors, note - alpha=0 -> error(priority) -> 1 for all samples in minibatch
            
        for ind, pi in zip(tree_indices, clipped_errors_alpha):
            self.tree.update(ind, pi)