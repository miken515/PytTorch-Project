def version_compare(version1, version2):
    # Insert your code here
    v1 = version1.split('.')
    v2 = version2.split('.')

    while len(v1) < len(v2):
        v1.append(str(0))

    for i in range(len(v1)):
        if v1[i] < v2[i]:
            return -1
        elif v1[i] > v2[i]:
            return 1
    return 0



print(version_compare(str("2.1.0"), str("2.0.1")))

'''
Input parameters: (str("2"), str("2.0"))
Result: (None,)
Expected: (0,)
---------------------------------
Input parameters: (str("2"), str("2.0.0"))
Result: (None,)
Expected: (0,)
---------------------------------
Input parameters: (str("2"), str("2.0.0.0"))
Result: (None,)
Expected: (0,)
---------------------------------
Input parameters: (str("2"), str("2.0.0.0.0"))
Result: (None,)
Expected: (0,)
---------------------------------
Input parameters: (str("2"), str("2.0.0.0.1"))
Result: (None,)
Expected: (-1,)
---------------------------------
Input parameters: (str("2"), str("2.1"))
Result: (None,)
Expected: (-1,)
---------------------------------
Input parameters: (str("2.1.0"), str("2.0.1"))
Result: (None,)
Expected: (1,)
---------------------------------
Input parameters: (str("2.10.0.1"), str("2.1.0.10"))
Result: (None,)
Expected: (1,)
---------------------------------
Input parameters: (str("2.0.1"), str("1.2000.1"))
Result: (None,)
Expected: (1,)

'''