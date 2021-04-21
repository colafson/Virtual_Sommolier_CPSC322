import matplotlib.pyplot as plt 
import numpy as np
import mysklearn.myutils as myutils
import scipy.stats as stats

#creates a pie chart
def makePieChart(x,y):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%",rotatelabels=True)
    plt.show()

#builds a bar graph
def makeBarGraph(x, y, xtick, x_label, y_label, title):
    plt.figure()
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x, xtick, rotation=90, horizontalalignment="right")
    plt.show()

#builds a histogram 
def makeHistogram(data, xlabel, ylabel,Title):
    plt.figure()
    plt.hist(data, bins=10)
    plt.title(Title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

#creates a scatterplot diagram 
def makeScatterPlot(x, y, xlabel, ylabel, title):
    plt.figure()
    plt.scatter(x,y)
    m, b, r, cov = myutils.compute_slope_intercept(x,y)
    xlabel += "\n Correlation: "+str(r)+"     Covariance: "+str(cov)
    plt.plot([min(x),max(x)], [m*min(x)+b, m*max(x)+b], c="r", lw=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


#creates a histogram with two different data sets
#mapped on top of one another
def makeDoubleHistogram(data1,data2, xlabel, ylabel, Title):
    plt.figure()
    plt.hist(data1, bins=10)
    plt.hist(data2, bins=10)
    plt.title(Title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()