import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vectors
x = [2, 1, 4]
y = [1, 3, 5]
z = [1, 2, 3]

def plotVectors(x, y, z, m,lx,ly,lz,path):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')

    # Colors and labels for each vector
    colors = ['r', 'g', 'b']
    labels = [str(round(lx))+"MPa", str(round(ly))+"MPa", str(round(lz))+"MPa"]

    # Plotting the vectors
    for i in range(len(x)):
        ax.quiver(0, 0, 0, x[i], y[i], z[i], color=colors[i], arrow_length_ratio=0.1, linewidth=1)
        # Adding labels to the vectors
        ax.text(x[i], y[i], z[i], labels[i], color=colors[i])

    # Setting the axis limits to be centered at (0, 0, 0) and extend from -m to +m
    ax.set_xlim([-m, m])
    ax.set_ylim([-m, m])
    ax.set_zlim([-m, m])
    ax.set_xlabel('N axis')
    ax.set_ylabel('E axis')
    ax.set_zlabel('D axis')
    ax.set_box_aspect([1,1,1])  # Ensuring a 1:1:1 aspect ratio
    ax.view_init(elev=45, azim=45)
    plt.savefig(path,dpi=600)
    return plt

# Example usage
#plotVectors(x, y, z, 2).show()
