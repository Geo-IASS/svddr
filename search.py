 #!/usr/bin/python
# -*- coding: utf-8 -*-

import os, numpy, operator

collection = "./wsj100/"
wordFile = open("vocab", "r").readlines()
query = ["oil", "industry"]
k = 50

# Term-document matrix
C = []
# (1) List words in the vocabulary file.
for document in os.listdir(collection):
    wordFreq = []
    fh = open(collection + document, "r")
    words = fh.readline().split(" ")
    for word in wordFile:
        stripped = word.strip()
        wordFreq.append(words.count(stripped))
    # (3) Create a list containing the lists from (2). Call it C (for collection).
    C.append(wordFreq)
# (4) Convert C to a numpy matrix object, i.e. C = numpy.matrix(C).
C = numpy.matrix(C)
# (5) Transpose C, i.e. C = C.T.
C = C.T

# Singular value decomposition
# C = UΣVT ≈ UkΣkVT, where k = k
U, s, VT = numpy.linalg.svd(C, full_matrices=False)
# Take the first k numbers in s and create a diagonal matrix
s_k = s[:k]
S_k = numpy.matrix(numpy.diag(s_k))
# Take the first k columns of U
U_k = U[:, :k]
# Take the first k rows of VT
VT_k = VT[:k, :]

q = []
for word in wordFile:
    val = (word.strip() in query)
    q.append(val)

q = numpy.matrix([q])
q = q.T

q_k = S_k.I * U_k.T * q
D_k = S_k * VT_k

inds = {}
for i in range(len(os.listdir(collection))):
    n = float(q_k.T * D_k[:, i])
    d = numpy.linalg.norm(q_k) * numpy.linalg.norm(D_k[:, i])
    inds[os.listdir(collection)[i]] =  n / d

newA = sorted(inds.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
for i in range(len(newA)):
    print newA[i][0]
