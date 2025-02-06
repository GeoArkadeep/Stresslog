from sympy import symbols, Eq, cos, sin, sqrt, solve

# Define variables
bhp, pp, tns, nu, theta = symbols("bhp pp tns nu theta")
Sb00, Sb11, Sb22, Sb01, Sb02, Sb12 = symbols("Sb00 Sb11 Sb22 Sb01 Sb02 Sb12")

# Derived variables
deltaP = bhp - pp
sigmaT = symbols("sigmaT")  # to be defined later
tensile_strength = tns

# Expressions for Szz, Stt, and Ttz
Szz = Sb22 - 2 * nu * (Sb00 - Sb11) * cos(2 * theta) - 4 * nu * Sb01 * sin(2 * theta)
Stt = Sb00 + Sb11 - 2 * (Sb00 - Sb11) * cos(2 * theta) - 4 * Sb01 * sin(2 * theta) - deltaP - sigmaT
Ttz = 2 * (Sb12 * cos(theta) - Sb02 * sin(theta))

# Expressions for STMax and Stmin
STMax = 0.5 * (Szz + Stt + sqrt((Szz - Stt)**2 + 4 * Ttz**2))
Stmin = 0.5 * (Szz + Stt - sqrt((Szz - Stt)**2 + 4 * Ttz**2))

# Set Stmin equal to tensile_strength
equation = Eq(Stmin, tensile_strength)

# Solve for bhp
bhp_critical_solution = solve(equation, bhp)
print(bhp_critical_solution)
"""
translated to standard python:
(40.0*Sb11**2*nu*np.cos(2.0*theta)**2 - 20.0*Sb11**2*nu*np.cos(2.0*theta) - 40.0*Sb11*Sb12*nu*np.sin(2.0*theta) + 80.0*Sb11*Sb12*nu*np.sin(4.0*theta) - 80.0*Sb11*Sb22*nu*np.cos(2.0*theta)**2 - 20.0*Sb11*Sb33*np.cos(2.0*theta) + 10.0*Sb11*Sb33 - 20.0*Sb11*nu*pp*np.cos(2.0*theta) + 20.0*Sb11*nu*sigmaT*np.cos(2.0*theta) - 2.0*Sb11*nu*ucs*np.cos(2.0*theta) - 2.0*Sb11*ucs*np.cos(2.0*theta) + Sb11*ucs + 160.0*Sb12**2*nu*np.sin(2.0*theta)**2 - 40.0*Sb12*Sb22*nu*np.sin(2.0*theta) - 80.0*Sb12*Sb22*nu*np.sin(4.0*theta) - 40.0*Sb12*Sb33*np.sin(2.0*theta) - 40.0*Sb12*nu*pp*np.sin(2.0*theta) + 40.0*Sb12*nu*sigmaT*np.sin(2.0*theta) - 4.0*Sb12*nu*ucs*np.sin(2.0*theta) - 4.0*Sb12*ucs*np.sin(2.0*theta) - 40.0*Sb13**2*np.sin(theta)**2 + 40.0*Sb13*Sb23*np.sin(2.0*theta) + 40.0*Sb22**2*nu*np.cos(2.0*theta)**2 + 20.0*Sb22**2*nu*np.cos(2.0*theta) + 20.0*Sb22*Sb33*np.cos(2.0*theta) + 10.0*Sb22*Sb33 + 20.0*Sb22*nu*pp*np.cos(2.0*theta) - 20.0*Sb22*nu*sigmaT*np.cos(2.0*theta) + 2.0*Sb22*nu*ucs*np.cos(2.0*theta) + 2.0*Sb22*ucs*np.cos(2.0*theta) + Sb22*ucs - 40.0*Sb23**2*np.cos(theta)**2 + 10.0*Sb33*pp - 10.0*Sb33*sigmaT + Sb33*ucs + pp*ucs - sigmaT*ucs + 0.1*ucs**2)/(-20.0*Sb11*nu*np.cos(2.0*theta) - 40.0*Sb12*nu*np.sin(2.0*theta) + 20.0*Sb22*nu*np.cos(2.0*theta) + 10.0*Sb33 + ucs)
"""