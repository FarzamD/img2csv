# %% imports
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from PIL import Image, ImageTk
import pytesseract
import pandas as pd

# %% defs
def cv2tk(image):
    # if im
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return ImageTk.PhotoImage(img)
def Edge(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    # smooth
    return cv2.medianBlur(edges,3)
# line detection
def line_threshold(shape, alternative=140):
    trh= min(shape)*2//5
    return max(trh, alternative)
def line_detection(image):
    return cv2.HoughLines(image, 1, np.pi/180, line_threshold(image.shape))
def plot_lines(lines, image):#array version of plot_lines
    img = image.copy()
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
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img
#corner detection
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
def corners(lines, image):
    lines_= lines.reshape((-1,2))
    alpha= np.pi/4
    lines_V_=lines_[lines_[:,1]<alpha]
    lines_H_=lines_[lines_[:,1]>alpha]

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
# table extraction
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
            roi= img[y0:y1, x0:x1]
            img_path= f'./cells/cell_{i}_{j}.png'
            cv2.imwrite(img_path, roi)
            text = pytesseract.image_to_string(roi)
            row.append(text.strip())
            # cmd='tesseract '+ img_path+ f' out_{i}_{j}'
            # subprocess.Popen(cmd)
        csv.append(row)
    return csv

# %% UI

class ImageUploaderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("table extractor")

        #tab control
        self.tabControl = ttk.Notebook(master)
        
        self.image_frame = ttk.Frame(self.tabControl)
        self.table_frame = ttk.Frame(self.tabControl)
        # tabs
        self.tabControl.add(self.image_frame, text="image")
        self.tabControl.add(self.table_frame, text="table")
        self.tabControl.pack(expand=1, fill="both")

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        upload_button = tk.Button(self.image_frame, text="Upload Image", command=self.upload_image)
        upload_button.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        if file_path:
            image = cv2.imread(file_path)
            img = cv2tk(image)
            self.image_label.config(image=img)
            self.image_label.image = img

            edges = Edge(image)
            lines= line_detection(edges)

            lines_plt= plot_lines(lines, image)            
            imgl= cv2tk(lines_plt)
            tk.Label(self.image_frame, text="detected lines:").pack()
            self.image_lines_label = tk.Label(self.image_frame)
            self.image_lines_label.pack()
            self.image_lines_label.config(image=imgl)
            self.image_lines_label.image = imgl

            p=corners(lines, image)
            csv= table(image, p)
            self.df= pd.DataFrame(csv)
            
            self.create_table_tab()
    def create_table_tab(self):
        #table
        tk.Label(self.table_frame, text="detected table:").pack()
        self.unpack_table()
        # confirm table and save
        save_button= tk.Button(self.table_frame, text="Confirm & Save table", command=self.save_table)
        save_button.pack(pady=20)

    def unpack_table(self):
        self.tree = ttk.Treeview(self.table_frame, columns=tuple(self.df.columns))
        for i in range(self.df.shape[0]):
            row= self.df.iloc[i,:]
            self.tree.insert("", "end", text=str(i+1), values=tuple(row.values))
        self.tree.pack(padx=50, pady=10)

    def save_table(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df.to_csv(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
