# img2csv

consists of three projects:
1. python image to csv
2. python image to csv with tkinter UI
3. image to csv electron app

+ python image to csv:
    + detects table lines
    + makes a table
    + extracts texts from image to cells

+ python image to csv with tkinter UI:
    + an app made with tkinter
    + added GUI to python app and contains python codes

+ image to csv electron app:
    + an app made with electron
    + added GUI to python app and contains some python code

 *** 
## python image to csv
### pre-requisites
+ install opencv
```linux
pip install cv2
```
or
```linux
apt-get install python-OpenCV
```
+ install Tesseract
+ install pytesseract
```linux
pip install pytesseract
```
### code blocks
#### edge detection
+ extract edges using cv2.Canny
    + smoothes edges using cv2.medianBlur
#### line detection
+ extracts lines from smoothed edges using cv2.HoughLines
    + split lines to horizontal and vertical based on line angle
#### table extraction
+ detects table corners from lines
+ extracts cells from table based on cell corners
    + turn cell images into text using pytesseract
+ merges cell texts into csv table

 ***
## python image to csv with tkinter UI
### pre-requisites
+ install opencv
    ```linux
    pip install cv2
    ```
    or
    ```linux
    apt-get install python-OpenCV
    ```
+ install Tesseract
+ install pytesseract
    ```linux
    pip install pytesseract
    ```
### code blocks
+ processing codes are the same as previous project
+ only UI code was added in this project
#### UI
+ UI code consists of two tabs:
    + image tab: where uploaded image and detected lines are displayed
    + table tab: where detected table is displayed and lets you choose as a file 
+ ***image tab:***
    ![app UI view](./blob/main/readme-blob/app.PNG)
    + first upload an image
    ![after uploading image](./blob/main/readme-blob/auto-detect-lines.PNG)
    + after uploading image if there are undetected lines or extra lines; modify number of lines
    + confirm number of lines to detect and create table in table_tab
    ![after uploading image](./blob/main/readme-blob/confirm-lines.PNG)
+ ***table tab:***

 ***
## image to csv electron app
### pre-requisites
```python
import pytesseract
text = pytesseract.image_to_string(img)
```