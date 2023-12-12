# %% imports & defs
# %%% imports
import cv2
import numpy as np

# %%% defs
def plot_lines(lines, image):#array version of plot_lines
    img = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    lines_r= lines.reshape((-1,2)).astype(np.float64)
    r, theta = lines_r[:,0], lines_r[:,1]

    a = np.cos(theta) # value of cos(theta)
    b = np.sin(theta) # value of sin(theta)
        
    x0 = a*r # value rcos(theta)
    y0 = b*r # value rsin(theta)

    x1_ = np.round(x0 + 1000*(-b)).astype(int) # rounded off value of (rcos(theta)-1000sin(theta))
    y1_ = np.round(y0 + 1000*(a)).astype(int) # rounded off value of (rsin(theta)+1000cos(theta))
    x2_ = np.round(x0 - 1000*(-b)).astype(int) # rounded off value of (rcos(theta)+1000sin(theta))
    y2_ = np.round(y0 - 1000*(a)).astype(int) # rounded off value of (rsin(theta)-1000cos(theta))
    for x1, y1, x2, y2 in zip(x1_, y1_, x2_, y2_):
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img
#
def line_threshold(shape, alternative=140):
    trh= min(shape)*2//5
    return max(trh, alternative)
#
# %% read image
img = cv2.imread('SIMPLE.PNG',cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('fullTable.png',cv2.IMREAD_GRAYSCALE)
image= img.copy()

# %% edge detection
edges = cv2.Canny(image, 50, 150, apertureSize=3)
# smooth
median = cv2.medianBlur(edges,3)
cv2.imshow('edges', edges)
cv2.imshow('Median Canny',median)


# %% line detection
lines = cv2.HoughLines(median, 1, np.pi/180, line_threshold(edges.shape))# This returns an array of r and theta values

del edges
# %%% construct & plot lines
image= img.copy()
l1= plot_lines(lines, image)
cv2.imshow('lines', l1)

cv2.waitKey(0)
cv2.destroyAllWindows()
# %% table extraction
# %%% corner extraction
lines_= lines.reshape((-1,2))
alpha= np.pi/4
lines_V=lines_[lines_[:,1]<alpha]
lines_H=lines_[lines_[:,1]>alpha]
del lines, median,l1, alpha

def m_c(r_theta):
    r, theta = r_theta    
    a = np.cos(theta) 
    b = np.sin(theta) 
    m=a/b
    c=b*r-r*a**2/b
    return m,c
def solve(r_theta1, r_theta2):
    if r_theta1[1]==0:
        x=r_theta1[0]
        m,c= m_c(r_theta2)
        return (x,m*x+c)        
    m1,c1= m_c(r_theta1)
    m2,c2= m_c(r_theta2)
    A=np.array([[-m1, 1],
                [-m2, 1]])
    b=np.array([[c1],[c2]])
    return (np.linalg.inv(A)@b).reshape((2,))
image=img.copy()
image= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
def corners(lines_V_, lines_H_, image):
    lines_V= lines_V_[lines_V_[:, 0].argsort()]
    lines_H= lines_H_[lines_H_[:, 0].argsort()]
    m,n= lines_V.shape[0], lines_H.shape[0]
    p=np.zeros((n,m,2), dtype=int)
    for i in range(n):
        rt2= lines_H[i]
        for j in range(m):
            rt1= lines_V[j]
            p[i,j]=solve(rt1, rt2)
            # g=255*i//n
            # r=255-255*j//m
            # x,y=p[i,j]
            # cv2.circle(image,(x,y),10, (0,g,r), -1)
    return p
p=corners(lines_V, lines_H, image)
# %%% table extraction
import pytesseract
import pandas as pd

def table(img, corners):
    p=corners.copy()
    n,m= p.shape[:2]
    csv=[]
    for i in range(n-1):
        row=[]
        for j in range(m-1):
            # print('j='+str(j))
            x0,y0=p[i,j]
            x1,y1=p[i+1,j+1]
            roi= image[y0:y1, x0:x1]
            img_path= f'./cells/cell_{i}_{j}.png'
            cv2.imwrite(img_path, roi)
            text = pytesseract.image_to_string(roi)
            row.append(text.strip())
            # cmd='tesseract '+ img_path+ f' out_{i}_{j}'
            # subprocess.Popen(cmd)
        csv.append(row)
    return csv
csv= table(image, p)
df= pd.DataFrame(csv)
