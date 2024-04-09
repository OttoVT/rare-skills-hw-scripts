from numpy import poly1d
from py_ecc.bn128 import G1, G2, pairing, add, multiply, eq, neg, G12, Z1, Z2
import numpy as np
from scipy.interpolate import lagrange
import galois
from py_ecc.fields.field_properties import (
    field_properties,
)
from functools import reduce
import random

# Test the pairing
assert pairing(multiply(G2, 2), G1) == pairing(G2, multiply(G1, 2)), "Not equal"

field_modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
#field_properties["bn128"]["field_modulus"]

GF = galois.GF(field_modulus)
# out = x⁴ - 5y²x²
# 1, out, x, y, v1, v2, v3
L = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, field_modulus-5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
])

R = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
])

O = np.array([
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, field_modulus-1, 0],
])

L_galois = GF(L)
R_galois = GF(R)
O_galois = GF(O)

x = GF(4)
y = GF(field_modulus-2)
v1 = x * x
v2 = v1 * v1         # x^4
v3 = GF(field_modulus-5)*y * y
out = v3*v1 + v2     # -5y^2 * x^2

witness = GF(np.array([1, out, x, y, v1, v2, v3]))

assert all(np.equal(np.matmul(L_galois, witness) * np.matmul(R_galois, witness), np.matmul(O_galois, witness))), "not equal"

def interpolate_column(col):
    xs = GF(np.array([1,2,3,4]))
    return galois.lagrange_poly(xs, col)

# axis 0 is the columns. apply_along_axis is the same as doing a for loop over the columns and collecting the results in an array
U_polys = np.apply_along_axis(interpolate_column, 0, L_galois)
V_polys = np.apply_along_axis(interpolate_column, 0, R_galois)
W_polys = np.apply_along_axis(interpolate_column, 0, O_galois)

def inner_product_polynomials_with_witness(polys, witness):
    mul_ = lambda x, y: x * y
    sum_ = lambda x, y: x + y
    return reduce(sum_, map(mul_, polys, witness))

term_1 = inner_product_polynomials_with_witness(U_polys, witness)
term_2 = inner_product_polynomials_with_witness(V_polys, witness)
term_3 = inner_product_polynomials_with_witness(W_polys, witness)

# t = (x - 1)(x - 2)(x - 3)(x - 4)
t = galois.Poly([1, field_modulus-1], field = GF) * galois.Poly([1, field_modulus-2], field = GF) * galois.Poly([1, field_modulus-3], field = GF) * galois.Poly([1, field_modulus-4], field = GF)

h = (term_1 * term_2 - term_3) // t

assert term_1 * term_2 == term_3 + h * t, "division has a remainder"

x = random.randint(1, 100)
term_1_x = term_1(x)
term_2_x = term_2(x)
term_3_x = term_3(x)
h_x = h(x)
t_x = t(x)
h_t_x = h_x * t_x
term_4_x = term_3_x + h_t_x
print(f"field_modulus: {field_modulus}")
print(f"x: {x}")
print(f"term_1_x: {term_1_x}")
print(f"term_2_x: {term_2_x}")
print(f"term_1_x * term_2_x: {term_1_x * term_2_x}")
print(f"term_4_x: {term_4_x}")

print(f"term_1_x * term_2_x: {int(term_1_x * term_2_x)}")
print(f"term_4_x: {int(term_4_x)}")

assert term_1_x * term_2_x == term_4_x, "Is not balanced"
assert pairing(multiply(G2, int(term_2_x)), multiply(G1, int(term_1_x))) == pairing(multiply(G2, 1), multiply(G1, int(term_4_x))), "Pairing not equal"

A = multiply(G1, int(term_1_x))
B = multiply(G2, int(term_2_x))
HT = multiply(G1, int(h_t_x))
C_1 = multiply(G1, int(term_3_x))
C_summ = add(C_1, HT)
C = multiply(G1, int(term_4_x))

assert C_summ == C, "C's are Not equal"

LEFT =  pairing(B, A)
RIGHT = pairing(neg(G2), C)

print(f"Computed LEFT: {LEFT}")
print(f"Computed RIGHT: {RIGHT}")

LHS = LEFT * RIGHT
print(f"Computed LHS: {LHS}")

assert LEFT == RIGHT, "Not equal"