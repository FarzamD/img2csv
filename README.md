# img2csv

consists of three projects:
1. python image to csv
2. python image to csv with tkinter UI
3. image to csv electron app

+ python image to csv:
    + takes an image
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
+ detects table from lines
+ extracts cells from table
    + turn cell images into text using pytesseract
+ merges cell texts into csv table

 ***
## python image to csv with tkinter UI


 ***
## image to csv electron app
### pre-requisites
```python
import pytesseract
text = pytesseract.image_to_string(img)

```
