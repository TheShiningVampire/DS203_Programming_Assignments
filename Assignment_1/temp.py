import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.cm as cm
import math

# Function to create a linked list from an array
def linked_list_from_array(array):
    head = None
    tail = None
    for i in range(len(array)):
        if i == 0:
            head = Node(array[i])
            tail = head
        else:
            tail.next = Node(array[i])
            tail = tail.next
    return head
    
