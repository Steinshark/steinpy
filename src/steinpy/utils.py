import numpy 
import math 
import json 
import os 
import errors



#   Takes arr, chunks it into *newlen* chunks,
#   and generates *new_arr* with the average of 
#   each chunk from *arr* 
def reduce_arr(arr:list,newlen:int):

    #Check for empty array
    if not arr:
        return []

    #Find GCF of len(arr) and len(newlen)
    gcf         = math.gcd(len(arr),newlen)
    mult_fact   = int(newlen / gcf) 
    div_fact    = int(len(arr) / gcf) 

    new_arr     = numpy.repeat(arr,mult_fact)

    return [sum(list(new_arr[n*div_fact:(n+1)*div_fact]))/div_fact for n in range(newlen)]



#   Returns the average of *arr* as defined as 
#   sum(arr) / len(arr)
def average(arr:list):

    #Check for empty array
    if not arr:
        return 0 
    
    return sum(arr) / len(arr)


