# medical-image-segmentation
A mini project of group 13 studying Intro to Deep Learning course at Hanoi University of Science &amp; Technology

<p align="center">
  <img src="assets/UIOverview.png" alt = "UI" title = "Overview of the application" width="500" height="270">
</p>

## Features
The project will allow you to classify lung radiography images into 3 types

* Normal (Lung is healthy)
* COVID (The person has covid)
* Non-COVID (The person doesn't have COVID but suffers different lung-related issues)

If a patient is diagnosed with COVID, the pipeline will segment the COVID-diagnosed radiography image further for better prescriptions and treatments

## Instructions to use

* Firstly, clone the repository's main branch into your desired directory using your git command prompt.

```git clone -b main https://github.com/king17pvp/medical-image-segmentation.git```

* Secondly, you can access the directory by this command.
* Thirdly, install required libraries via requirement.txt
```pip install -r requirements.txt```
* Finally, run the project by 

```python app.py```
## 
