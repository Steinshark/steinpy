import chess
import torch 
import numpy 
import random
import time
import string

#The Node must be able to:
#   Contain info on parents, children, board fen, turn, current score, p,v, n_visits
class Node:
    def __init__(self,board_fen:str,turn:bool):
        self.parent     = parent
        self.children   = {}
        self.board_fen  = board_fen
        self.turn       = turn
        self.p          = 0 
        self.v          = 0 
        self.n_visits   = 0
        
        self.node_name  = "".join(random.choices('123456789',10))

#The tree must be able to:
#   Search downwards iteratively 
#   Keep track of nodes reached on the way down
#   Update nodes on the way up  
class Tree:

    def __init__(self):
        self.root_node  = Node()



if __name__ == "__main__":
    # Definition for singly-linked list.

    def print_list(node):
        empty_str   = ""
        while node.next:
            empty_str += f"{node.val}->"
            node = node.next 
        empty_str += str(node.val)
        return empty_str[:100]
        
    #Definition for a binary tree node.
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    class Solution:
        def num_b_trees(arr:list[int]):
            from itertools import 




sol     = Solution()
print(sol.kthGrammar(2,1))