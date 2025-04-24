import random
import numpy as np
from sympy import symbols, Eq, cos, sin, sqrt, solve, expand, Add, N

# Define symbols (using the same names as in your equation)
bhp, pp, ucs, nu, theta = symbols("bhp pp ucs nu theta")
Sb00, Sb11, Sb22, Sb01, Sb02, Sb12 = symbols("Sb00 Sb11 Sb22 Sb01 Sb02 Sb12")
sigmaT = symbols("sigmaT")  # defined later

# Derived variables and expressions (as in your code)
deltaP = bhp - pp
tensile_strength = -ucs / 10

Szz = Sb22 - 2*nu*(Sb00 - Sb11)*cos(2*theta) - 4*nu*Sb01*sin(2*theta)
Stt = Sb00 + Sb11 - 2*(Sb00 - Sb11)*cos(2*theta) - 4*Sb01*sin(2*theta) - deltaP - sigmaT
Ttz = 2*(Sb12*cos(theta) - Sb02*sin(theta))

STMax = 0.5*(Szz + Stt + sqrt((Szz - Stt)**2 + 4*Ttz**2))
Stmin = 0.5*(Szz + Stt - sqrt((Szz - Stt)**2 + 4*Ttz**2))

# The equation that defines the critical condition
equation = Eq(Stmin, tensile_strength)

# Solve for bhp (this returns a list of solutions)
bhp_critical_solution = solve(equation, bhp)
print(bhp_critical_solution)
"""[(40.0*Sb00**2*nu*cos(2.0*theta)**2 - 20.0*Sb00**2*nu*cos(2.0*theta) - 40.0*Sb00*Sb01*nu*sin(2.0*theta) + 80.0*Sb00*Sb01*nu*sin(4.0*theta) - 80.0*Sb00*Sb11*nu*cos(2.0*theta)**2 - 20.0*Sb00*Sb22*cos(2.0*theta) + 10.0*Sb00*Sb22 - 20.0*Sb00*nu*pp*cos(2.0*theta) + 20.0*Sb00*nu*sigmaT*cos(2.0*theta) - 2.0*Sb00*nu*ucs*cos(2.0*theta) - 2.0*Sb00*ucs*cos(2.0*theta) + Sb00*ucs + 160.0*Sb01**2*nu*sin(2.0*theta)**2 - 40.0*Sb01*Sb11*nu*sin(2.0*theta) - 80.0*Sb01*Sb11*nu*sin(4.0*theta) - 40.0*Sb01*Sb22*sin(2.0*theta) - 40.0*Sb01*nu*pp*sin(2.0*theta) + 40.0*Sb01*nu*sigmaT*sin(2.0*theta) - 4.0*Sb01*nu*ucs*sin(2.0*theta) - 4.0*Sb01*ucs*sin(2.0*theta) - 40.0*Sb02**2*sin(theta)**2 + 40.0*Sb02*Sb12*sin(2.0*theta) + 40.0*Sb11**2*nu*cos(2.0*theta)**2 + 20.0*Sb11**2*nu*cos(2.0*theta) + 20.0*Sb11*Sb22*cos(2.0*theta) + 10.0*Sb11*Sb22 + 20.0*Sb11*nu*pp*cos(2.0*theta) - 20.0*Sb11*nu*sigmaT*cos(2.0*theta) + 2.0*Sb11*nu*ucs*cos(2.0*theta) + 2.0*Sb11*ucs*cos(2.0*theta) + Sb11*ucs - 40.0*Sb12**2*cos(theta)**2 + 10.0*Sb22*pp - 10.0*Sb22*sigmaT + Sb22*ucs + pp*ucs - sigmaT*ucs + 0.1*ucs**2)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs)]"""

# Pick the first solution (if there are multiple)
bhp_expr = bhp_critical_solution[0]
# Expand the expression into a sum of terms
expanded_expr = expand(bhp_expr)

# Break the expanded expression into its additive terms.
terms = Add.make_args(expanded_expr)
print("The expression has been broken into {} terms.".format(len(terms)))

# We'll store for each term a list of contributions from each sample.
# The key is the string version of the term.
term_stats = {str(term): [] for term in terms}

num_samples = 1000

# Loop over random samples
for i in range(num_samples):
    # Generate a random substitution dictionary. Adjust ranges as needed.
    subs = {
        pp: random.uniform(1, 100),
        ucs: random.uniform(1, 100),
        nu: random.uniform(0.2, 0.35),  # typical Poisson ratio range
        theta: random.uniform(0, 360),
        Sb00: random.uniform(0, 100),
        Sb11: random.uniform(0, 100),
        Sb22: random.uniform(0, 100),
        Sb01: random.uniform(0, 100),
        Sb02: random.uniform(0, 100),
        Sb12: random.uniform(0, 100),
        sigmaT: random.uniform(0, 100)
    }
    
    # Compute the absolute value of each term
    term_values = {}
    total_val = 0.0
    for term in terms:
        # Evaluate the term numerically for this sample.
        term_val = abs(N(term, subs=subs))
        term_values[str(term)] = term_val
        total_val += term_val

    # Skip this sample if the total is (close to) zero
    if total_val == 0:
        continue

    # Compute relative contribution for each term and store it
    for term_str, val in term_values.items():
        relative_contrib = val / total_val  # as a fraction (0 to 1)
        term_stats[term_str].append(relative_contrib)

# Compute average, minimum, and maximum contributions (in percentages) for each term.
summary_stats = {}
for term_str, contributions in term_stats.items():
    if contributions:  # avoid division by zero if no contributions recorded
        avg = 100 * np.mean(contributions)
        min_val = 100 * np.min(contributions)
        max_val = 100 * np.max(contributions)
    else:
        avg = min_val = max_val = 0.0
    summary_stats[term_str] = (avg, min_val, max_val)

# Sort terms by average contribution (largest first)
sorted_terms = sorted(summary_stats.items(), key=lambda x: x[1][0], reverse=True)

# Print the results
print("\nAverage, Minimum, and Maximum percentage contributions of each term:")
for term_str, (avg, min_val, max_val) in sorted_terms:
    print(f"{term_str:80s}: Average = {avg:6.2f}%, Min = {min_val:6.2f}%, Max = {max_val:6.2f}%")
"""
The expression has been broken into 38 terms.

Average, Minimum, and Maximum percentage contributions of each term:
-40.0*Sb12**2*cos(theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   9.39%, Min =   0.00%, Max =  69.40%
-40.0*Sb02**2*sin(theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   8.21%, Min =   0.00%, Max =  85.69%
40.0*Sb02*Sb12*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   7.97%, Min =   0.00%, Max =  46.34%
160.0*Sb01**2*nu*sin(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   7.82%, Min =   0.00%, Max =  58.35%
-40.0*Sb01*Sb22*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   7.16%, Min =   0.00%, Max =  31.87%
80.0*Sb00*Sb01*nu*sin(4.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   4.10%, Min =   0.00%, Max =  22.78%
-80.0*Sb01*Sb11*nu*sin(4.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   4.09%, Min =   0.00%, Max =  20.15%
20.0*Sb11*Sb22*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   4.09%, Min =   0.00%, Max =  25.79%
-20.0*Sb00*Sb22*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   4.06%, Min =   0.00%, Max =  23.20%
10.0*Sb22*pp/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.68%, Min =   0.01%, Max =  32.80%
-80.0*Sb00*Sb11*nu*cos(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.53%, Min =   0.00%, Max =  24.62%
-10.0*Sb22*sigmaT/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.41%, Min =   0.00%, Max =  28.63%
10.0*Sb00*Sb22/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.19%, Min =   0.01%, Max =  23.56%
10.0*Sb11*Sb22/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.13%, Min =   0.01%, Max =  22.10%
40.0*Sb11**2*nu*cos(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.56%, Min =   0.00%, Max =  27.51%
40.0*Sb00**2*nu*cos(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.49%, Min =   0.00%, Max =  33.39%
-40.0*Sb01*nu*pp*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.27%, Min =   0.00%, Max =  14.16%
40.0*Sb01*nu*sigmaT*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.07%, Min =   0.00%, Max =  12.33%
-40.0*Sb00*Sb01*nu*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.01%, Min =   0.00%, Max =  10.95%
-40.0*Sb01*Sb11*nu*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.97%, Min =   0.00%, Max =  11.91%
20.0*Sb11**2*nu*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.59%, Min =   0.00%, Max =  13.81%
-20.0*Sb00**2*nu*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.56%, Min =   0.00%, Max =  17.68%
20.0*Sb11*nu*pp*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.32%, Min =   0.00%, Max =   8.94%
-20.0*Sb00*nu*pp*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.32%, Min =   0.00%, Max =  16.03%
-20.0*Sb11*nu*sigmaT*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.25%, Min =   0.00%, Max =  13.39%
20.0*Sb00*nu*sigmaT*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.24%, Min =   0.00%, Max =  16.95%
-4.0*Sb01*ucs*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.88%, Min =   0.00%, Max =   4.60%
-2.0*Sb00*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.51%, Min =   0.00%, Max =   6.40%
2.0*Sb11*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.50%, Min =   0.00%, Max =   3.67%
pp*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.46%, Min =   0.02%, Max =   5.25%
-sigmaT*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.41%, Min =   0.00%, Max =   3.34%
Sb00*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.40%, Min =   0.00%, Max =   3.21%
Sb11*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.39%, Min =   0.00%, Max =   2.86%
Sb22*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.38%, Min =   0.00%, Max =   2.46%
-4.0*Sb01*nu*ucs*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.24%, Min =   0.00%, Max =   1.22%
-2.0*Sb00*nu*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.14%, Min =   0.00%, Max =   1.61%
2.0*Sb11*nu*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.14%, Min =   0.00%, Max =   0.96%
0.1*ucs**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.06%, Min =   0.00%, Max =   0.62%
"""
"""
Average, Minimum, and Maximum percentage contributions of each term:
-40.0*Sb12**2*cos(theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   9.13%, Min =   0.00%, Max =  79.69%
-40.0*Sb02**2*sin(theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   8.73%, Min =   0.00%, Max =  67.05%
160.0*Sb01**2*nu*sin(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   8.37%, Min =   0.00%, Max =  55.13%
40.0*Sb02*Sb12*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   8.09%, Min =   0.00%, Max =  41.39%
-40.0*Sb01*Sb22*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   7.31%, Min =   0.00%, Max =  35.73%
20.0*Sb11*Sb22*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   4.18%, Min =   0.00%, Max =  25.24%
-20.0*Sb00*Sb22*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   4.13%, Min =   0.00%, Max =  24.01%
-80.0*Sb01*Sb11*nu*sin(4.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.90%, Min =   0.00%, Max =  18.85%
80.0*Sb00*Sb01*nu*sin(4.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.89%, Min =   0.00%, Max =  25.32%
-10.0*Sb22*sigmaT/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.47%, Min =   0.00%, Max =  25.61%
-80.0*Sb00*Sb11*nu*cos(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.46%, Min =   0.00%, Max =  27.02%
10.0*Sb22*pp/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.43%, Min =   0.00%, Max =  24.68%
10.0*Sb00*Sb22/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.29%, Min =   0.00%, Max =  18.24%
10.0*Sb11*Sb22/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   3.25%, Min =   0.00%, Max =  23.50%
40.0*Sb11**2*nu*cos(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.56%, Min =   0.00%, Max =  22.29%
40.0*Sb00**2*nu*cos(2.0*theta)**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.47%, Min =   0.00%, Max =  25.46%
40.0*Sb01*nu*sigmaT*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.10%, Min =   0.00%, Max =  13.60%
-40.0*Sb01*nu*pp*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.09%, Min =   0.00%, Max =  14.46%
-40.0*Sb00*Sb01*nu*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.04%, Min =   0.00%, Max =  13.48%
-40.0*Sb01*Sb11*nu*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   2.00%, Min =   0.00%, Max =  11.98%
20.0*Sb11**2*nu*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.60%, Min =   0.00%, Max =  11.49%
-20.0*Sb00**2*nu*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.57%, Min =   0.00%, Max =  12.73%
-20.0*Sb00*nu*pp*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.26%, Min =   0.00%, Max =  12.20%
20.0*Sb11*nu*pp*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.20%, Min =   0.00%, Max =  10.50%
-20.0*Sb11*nu*sigmaT*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.19%, Min =   0.00%, Max =   9.94%
20.0*Sb00*nu*sigmaT*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   1.16%, Min =   0.00%, Max =   9.49%
-4.0*Sb01*ucs*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.84%, Min =   0.00%, Max =   6.14%
-2.0*Sb00*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.46%, Min =   0.00%, Max =   3.82%
2.0*Sb11*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.45%, Min =   0.00%, Max =   4.81%
-sigmaT*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.38%, Min =   0.00%, Max =   3.90%
pp*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.38%, Min =   0.00%, Max =   3.75%
Sb22*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.37%, Min =   0.00%, Max =   5.59%
Sb00*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.37%, Min =   0.00%, Max =   3.44%
Sb11*ucs/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.35%, Min =   0.00%, Max =   3.36%
-4.0*Sb01*nu*ucs*sin(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.23%, Min =   0.00%, Max =   1.31%
-2.0*Sb00*nu*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.13%, Min =   0.00%, Max =   1.20%
2.0*Sb11*nu*ucs*cos(2.0*theta)/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.12%, Min =   0.00%, Max =   1.12%
0.1*ucs**2/(-20.0*Sb00*nu*cos(2.0*theta) - 40.0*Sb01*nu*sin(2.0*theta) + 20.0*Sb11*nu*cos(2.0*theta) + 10.0*Sb22 + ucs): Average =   0.05%, Min =   0.00%, Max =   0.59%
"""