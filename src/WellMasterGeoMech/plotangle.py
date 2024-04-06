import matplotlib.pyplot as plt
import numpy as np

def plotfracs(data):
    x = data[:, 1]
    x2 = x+180
    y = data[:, 0]
    angles = data[:, 2]
    angles2 = -data[:, 3]
    # Creating the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 360)  # Adjusting x-axis limit to fit your data
    #ax.set_ylim(2450, 2460)  # Adjusting y-axis limit to fit your data
    #ax.set_aspect(0.2)  # This makes 1 unit in x equal to 1 unit in y

    # Plotting each line segment based on the angle
    for x_i, y_i, angle in zip(x, y, angles):
        # Calculating the end point of the line segment
        # Assuming a fixed length for each segment
        length = 1  # Adjusted length to be visible given the scale of your data
        end_x = x_i + length * np.cos(np.radians(angle))
        end_y = y_i + length * np.sin(np.radians(angle))
        
        # Plotting the line segment
        ax.plot([x_i, end_x], [y_i, end_y], linewidth=0.01, color='black')  # Added markers for clarity
        
        
    for x2_i, y_i, angle2 in zip(x2, y, angles2):
        # Calculating the end point of the line segment
        # Assuming a fixed length for each segment
        length = 1  # Adjusted length to be visible given the scale of your data
        end_x = x2_i - length * np.cos(np.radians(angle2))
        end_y = y_i - length * np.sin(np.radians(angle2))
        
        # Plotting the line segment
        ax.plot([x2_i, end_x], [y_i, end_y], linewidth=0.01, color='black')  # Added markers for clarity
        
    return plt
#plt.show()

def plotfracsQ(data):
    x = data[:, 1]
    x2 = x+180
    y = data[:, 0]
    angles = data[:, 2]
    angles2 = -data[:, 3]
    # Creating the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 360)  # Adjusting x-axis limit to fit your data
    #ax.set_ylim(2450, 2460)  # Adjusting y-axis limit to fit your data
    #ax.set_aspect(0.2)  # This makes 1 unit in x equal to 1 unit in y

    return plt
#plt.show()


def plotfrac(data):
    tvd,fr,angles,minangle,maxangle = data
    dia = 8.5 #inches, bit
    circumference = np.pi*dia #in inches
    cm = 0.0254*circumference
    i = 0
    d = np.zeros(360)
    yj = np.zeros(360)
    depths = np.zeros(360)
    sign = np.zeros(360)
    midpoint1 = min(minangle+90,(minangle+270)%360)
    midpoint2 = max(minangle+90,(minangle+270)%360)
    spVa = 360
    spVb = 0
    while(i<360):
        if i>midpoint1 and i<midpoint2:
            d[i] = (i-(minangle+180))
        else:
            if i<midpoint1:
                d[i] = i-minangle
            else:
                d[i] = i-(360+minangle)
        if abs(d[i])==270:
            d[i]=90*(abs(d[i])/d[i])
        yj[i] = (np.tan(np.radians(angles[i]))*d[i]) #FlipVertN if abs
        sign[i] = (np.tan(np.radians(angles[i]))*d[i])/abs(np.tan(np.radians(angles[i]))*d[i])
        depths[i] = (((yj[i]-180)/180)*(cm/2))
        if d[i-1]==0:
            yj[i-1] = (yj[i-2]+yj[i])/2
            spV =  yj[i-1]
            depths[i-1] = (depths[i-2]+depths[i])/2
            spVa = min(depths[i-1],spVa)
            spVb = max(depths[i-1],spVb)

        i+=1
    
    yj = yj-spV
    #depths = depths*sign #Dont FlipV if signed #AntiFlipV
    plt.figure(figsize=(10, 10))
    plt.plot(yj)
    #Setting axis limits
    plt.xlim(0, 360)
    #plt.ylim(-180, 180)
    plt.savefig('frac.png')
    print(yj)
    yj[(maxangle-10)%360:(maxangle+15)%360]=np.nan
    yj[(maxangle+170)%360:(maxangle+195)%360]=np.nan
    #depths[(maxangle-10)%360:(maxangle+15)%360]=np.nan
    #depths[(maxangle+170)%360:(maxangle+195)%360]=np.nan
    i=0
    cyj = yj
    cyj[midpoint1:midpoint2] = cyj[midpoint1:midpoint2][::-1]
    avdepths = depths.copy()
    while i<360: #ShiftVert1
        #if i<midpoint1 or i>midpoint2-1:
        #    depths[i] = depths[i]-spVa
        #else:
        #    depths[i] = depths[i]-spVb
        if abs(depths[i])>1.6:
            #depths[i] = np.nan
            avdepths[i] = np.nan
        if fr[i]<1:
            avdepths[i] = np.nan
        i+=1
    av1 = np.nanmean(np.concatenate([depths[0:(midpoint1-1)],depths[midpoint2:360]]))
    av2 = np.nanmean(depths[midpoint1:(midpoint2-1)])
    i=0
    while i<360: #ShiftVert2
        if i<midpoint1 or i>midpoint2-1:
            depths[i] = depths[i]-av1
        else:
            depths[i] = depths[i]-av2
        i+=1
    
    i=0
    fdepths=depths.copy()
    print(fdepths)
    deldep = (np.nanmean(depths)) 
    depths = depths-deldep
    fdepths = fdepths-deldep
    cdepths = depths+tvd
    while i<360:
        if abs(depths[i])>1.6:
            cdepths[i] = np.nan
        if fr[i]<1:
            cdepths[i] = np.nan
        i+=1
    fdepths = fdepths+tvd
    #cdepths[midpoint1:midpoint2] = cdepths[midpoint1:midpoint2][::-1]# FlipHorzR
    #fdepths[midpoint1:midpoint2] = fdepths[midpoint1:midpoint2][::-1]# FlipHorzR
    print(fdepths)
    print(d)
    
    """
    while i<360:
        if fr[i]>0:
            plt.scatter(i, yj[i], color='black', marker='o')
        i+=1
    """

    return cdepths,fdepths
#plt.show()
