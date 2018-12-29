import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
ax.set_xlim(0.5, 4.5)
plt.figure(figsize=(1,1))
plt.show()



ax.bar() 	#Vertical rectangles
ax.barh() 	#Horizontal rectangles
ax.axhline() 	#Horizontal line across axes
ax.vline() 	#Vertical line across axes
ax.fill() 	#Filled polygons
ax.fill_between() 	#Fill between y-values and 0
ax.stackplot() 	#Stack plot


ax.arrow() 	#Arrow
ax.quiver() 	#2D field of arrows
ax.streamplot() 	#2D vector fields
ax.hist() 	#Histogram
ax.boxplot() 	#Boxplot
ax.violinplot() 	#Violinplot