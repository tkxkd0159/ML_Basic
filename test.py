# from math import gcd

def is_array(A):

    gcd = A[0]

    for a in A[1:]:
        while a:
            gcd, a = a, gcd % a

    return gcd


print(is_array([9, 18]))
