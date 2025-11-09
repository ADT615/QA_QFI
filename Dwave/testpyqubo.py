from pyqubo import Spin
s1, s2, s3, s4 = Spin("s1"), Spin("s2"), Spin("s3"), Spin("s4")
H = (4*s1 + 2*s2 + 7*s3 + s4)**2
model = H.compile()
qubo, offset = model.to_qubo()
print(qubo) # doctest: +SKIP
# print(offset)