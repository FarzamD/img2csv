# %% imports
from cv2 import cvtColor, COLOR_BGR2RGB, Canny, medianBlur, HoughLines
from cv2 import line, circle, imwrite, imread
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from pytesseract import image_to_string as tess_img2str
from pandas import DataFrame as DF
from PIL import Image, ImageTk

# %% defs
def cv2tk(image):
    img = cvtColor(image, COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return ImageTk.PhotoImage(img)
def Edge(image):
    edges = Canny(image, 50, 150, apertureSize=3)
    return medianBlur(edges,3) # return smoothed edges

# line detection
def line_threshold(shape, alternative=140):
    trh= min(shape)*2//5
    return max(trh, alternative)
def line_detection(image,alternative=140):
    return HoughLines(image, 1, np.pi/180, line_threshold(image.shape,alternative=alternative))
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
        line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
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
            # B,G,R = 0, 255*i//n, 255-255*j//m
            # x,y=p[i,j]
            # circle(image,(x,y),10, (B,G,R), -1)
    return p

# table extraction
def table(img, corners):
    p=corners.copy()
    n,m= p.shape[:2]
    csv=[]
    for i in range(n-1):
        row=[]
        for j in range(m-1):
            x0,y0=p[i,j]
            x1,y1=p[i+1,j+1]
            roi= img[y0:y1, x0:x1]
            img_path= f'./cells/cell_{i}_{j}.png'
            imwrite(img_path, roi)
            text = tess_img2str(roi)
            row.append(text.strip())
        csv.append(row)
    return csv

# %% UI

class ImageUploaderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("table extractor")
        self.alt=140

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
            self.image = imread(file_path)
            img = cv2tk(self.image)
            self.image_label.config(image=img)
            self.image_label.image = img

            self.edges = Edge(self.image)

            tk.Label(self.image_frame, text="detected lines:").pack()

            self.lines= line_detection(self.edges)
            lines_plt= plot_lines(self.lines, self.image)            
            self.imgl= cv2tk(lines_plt)

            self.image_lines_label = tk.Label(self.image_frame)
            self.image_lines_label.pack()
            self.show_lines()
            
            button_frame = tk.Frame(self.image_frame)
            button_frame.pack()

            incp_button= tk.Button(button_frame, text="detect way more lines", command=self.morep_lines)
            incp_button.pack(pady=20, padx=20, side=tk.LEFT)
            inc_button= tk.Button(button_frame, text="detect more lines", command=self.more_lines)
            # inc_button.pack(row=4, column=0, pady=20, padx=20)
            inc_button.pack(pady=20, padx=20, side=tk.LEFT)
            dec_button= tk.Button(button_frame, text="detect less lines", command=self.less_lines)
            # dec_button.pack(row=4, column=1, pady=20, padx=20)
            dec_button.pack(pady=20, padx=20, side=tk.LEFT)
            dec_button= tk.Button(self.image_frame, text="confirm lines", command=self.conf_lines)
            # dec_button.pack(row=4, column=1, pady=20, padx=20)
            dec_button.pack(pady=20, padx=20,side=tk.TOP)

            self.create_df()
            self.create_table_tab()
    def create_table_tab(self):
        #table
        self.det_tables_lbl= tk.Label(self.table_frame, text="detected table:")
        self.det_tables_lbl.pack()
        self.unpack_table()
        # confirm table and save
        self.save_button= tk.Button(self.table_frame, text="Confirm & Save table", command=self.save_table)
        self.save_button.pack(pady=20)

    def create_df(self):
        p=corners(self.lines, self.image)
        csv= table(self.image, p)
        self.df= DF(csv)
        

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
    def show_lines(self):
        self.image_lines_label.config(image=self.imgl)
        self.image_lines_label.image = self.imgl
    def less_lines(self):
        self.alt+=2
        self.lines= line_detection(self.edges, alternative=self.alt)
        lines_plt= plot_lines(self.lines, self.image)            
        self.imgl= cv2tk(lines_plt)
        self.show_lines()
    def more_lines(self):
        self.alt-=3
        self.lines= line_detection(self.edges, alternative=self.alt)
        lines_plt= plot_lines(self.lines, self.image)            
        self.imgl= cv2tk(lines_plt)
        self.show_lines()
    def morep_lines(self):
        self.alt-=10
        self.lines= line_detection(self.edges, alternative=self.alt)
        lines_plt= plot_lines(self.lines, self.image)            
        self.imgl= cv2tk(lines_plt)
        self.show_lines()
    def conf_lines(self):
        self.create_df()
        
        self.det_tables_lbl.destroy()
        self.save_button.destroy()
        self.tree.destroy()

        self.create_table_tab()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
