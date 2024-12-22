"""
Copyright (c) 2024-2025 ROCK LAB PRIVATE LIMITED
This file is part of "Stresslog" project and is released under the 
GNU Affero General Public License v3.0 (AGPL-3.0)
See the GNU Affero General Public License for more details: <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vectors
x = [2, 1, 4]
y = [1, 3, 5]
z = [1, 2, 3]

def plotVectors(x, y, z, m,lx,ly,lz):
    
    #x += [2, 0, 0]
    #y += [0, 2, 0]
    #z += [0, 0, 2]

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Original colors and labels for each vector
    original_colors = ['r', 'g', 'b']
    original_labels = [
        f"{round(lx)}MPa", 
        f"{round(ly)}MPa", 
        f"{round(lz)}MPa"
    ]

    # Plotting the original vectors
    for i in range(len(x)):
        ax.quiver(0, 0, 0, x[i], y[i], z[i], color=original_colors[i], arrow_length_ratio=0.1, linewidth=1)
        ax.text(x[i], y[i], z[i], original_labels[i], color=original_colors[i])

    # Unit vectors (N, E, D)
    unit_vectors = [(0.5, 0, 0), (0,0.5, 0), (0, 0, 0.5)]
    unit_colors = ['k', 'k', 'k']  # 'k' for black for the unit vectors
    unit_labels = ["N", "E", "D"]
    
    # Plotting the original vectors
    for i in range(len(x)):
        ax.quiver(0, 0, 0, -x[i], -y[i], -z[i], color=original_colors[i], arrow_length_ratio=0.1, linewidth=1)
        ax.text(x[i], y[i], z[i], original_labels[i], color=original_colors[i])

    # Unit vectors (N, E, D)
    unit_vectors = [(0.5, 0, 0), (0,0.5, 0), (0, 0, 0.5)]
    unit_colors = ['k', 'k', 'k']  # 'k' for black for the unit vectors
    unit_labels = ["N", "E", "D"]

    # Plotting the unit vectors
    for i, (ux, uy, uz) in enumerate(unit_vectors):
        ax.quiver(0, 0, 0, ux, uy, uz, color=unit_colors[i], arrow_length_ratio=0.1, linewidth=1)
        ax.text(ux, uy, uz, unit_labels[i], color=unit_colors[i])

    ax.set_xlim([-m, m])
    ax.set_ylim([-m, m])
    ax.set_zlim([-m, m])
    ax.set_xlabel('N axis')
    ax.set_ylabel('E axis')
    ax.set_zlabel('D axis')
    
    unit_vectors = [(-0.5, 0, 0), (0,-0.5, 0), (0, 0, -0.5)]
    unit_colors = ['k', 'k', 'k']  # 'k' for black for the unit vectors
    unit_labels = ["S", "W", "U"]
    
    # Plotting the unit vectors
    for i, (ux, uy, uz) in enumerate(unit_vectors):
        ax.quiver(0, 0, 0, ux, uy, uz, color=unit_colors[i], arrow_length_ratio=0.1, linewidth=1)
        ax.text(ux, uy, uz, unit_labels[i], color=unit_colors[i])

    ax.set_xlim([-m, m])
    ax.set_ylim([-m, m])
    ax.set_zlim([-m, m])
    ax.set_xlabel('N axis')
    ax.set_ylabel('E axis')
    ax.set_zlabel('D axis')
    
    
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=45, azim=45)
    #plt.savefig(path, dpi=600)
    #plt.show()
    return plt

# Example usage
#plotVectors(x, y, z, 2, 23,30,40).show()
def showvec(x, y, z, m,lx,ly,lz,path):
    plotVectors(x, y, z, m,lx,ly,lz).show()
    return
def savevec(x, y, z, m,lx,ly,lz,path):
    plt = plotVectors(x, y, z, m,lx,ly,lz)
    plt.savefig(path, dpi=600)
    plt.close()
    return
