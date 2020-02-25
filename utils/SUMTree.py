import numpy as np

class SUMTree:
    '''
    Implementation using Binary Tree approach using array.
    To-Do == Implement using other data structure?
    Couple of things to make a note of,
    - Data Array ->  experience replay buffer
    - tree       ->  binary sum tree for efficient storage, access, search
    - If Data array size is 'n', the #leaf nodes in the tree would be n. In this case, the total #nodes in tree would be 2n-1
    - If data array index is 'x', this corresponds to the leaf node number in tree (starting from left). So, tree index will be x+n-1
    - If leaf node index is 'y', data array index will be y+1-n
    '''
    
    def __init__(self, capacity):
        '''
        
        '''
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0                               # Pointer to experience in data array

    def insert(self, priority, experience):
        '''
        Insert the tree with priority and add the experience at the corresponding index.
        '''
        tree_index = self.data_pointer + self.capacity - 1  # Get tree index from data array index
        self.update(tree_index, priority)                   # update tree with priority value - propogate the change from new addition
        self.data[self.data_pointer] = experience           # add experience to the data array
        self.data_pointer += 1                              # increment pointer to data array to keep a track
        if(self.data_pointer>=self.capacity):               # reset when overflow
            self.data_pointer = 0

    def update(self, tree_index, priority):
        '''
        Change the existing priority value at tree_index to new value=priority
        '''
        delta = priority - self.tree[tree_index]            # propogate this change throughout the tree
        self.tree[tree_index] = priority                    # assign the new value
        while tree_index!=0:                                # tree_index = 0 will be the last update
            tree_index = (tree_index - 1) // 2              # floor round off to get to parent
            self.tree[tree_index] += delta                  # add the delta every parent connected to initial tree index

    def get_leaf(self, value):
        '''
        Here, priority value is passed and its index needs to be retrieved to access the corresponding experience.
        Heuristic of moving towards left sub-tree for val<=node.left.value is applied. The opposite, where smaller elements are traversed towards right sub-tree is equally valid.
        '''
        index = 0
        while 1:
            left_child_index = 2*index + 1                  # since sum tree constructed is a binary tree
            right_child_index = 2*index + 2                 # since sum tree constructed is a binary tree

            if left_child_index >= 2*self.capacity-1:       # when left/right pointer goes out of bound. This happens when index reaches last layer.
                leaf_index = index                          # since index is in the last layer, it corresponds to a leaf node -> priority value
                break
            else:
                if self.tree[left_child_index] >= value:    # This is a heuristic to move smaller values to the left sub-tree.
                    index = left_child_index
                else:
                    value -= self.tree[left_child_index]    # Subsequently, move right for higher values
                    index = right_child_index   

        array_index = leaf_index + 1 - self.capacity

        return self.data[array_index], self.tree[leaf_index], leaf_index
        
    def total(self):
        '''
        This function returns the total sum of the elements stored in the tree i.e. totakl priority value of the tree
        '''
        return self.tree[0]