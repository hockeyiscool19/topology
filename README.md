# Introduction

This repository automates making design for CNC machines. That is, you choose a desired state to make a map. You choose a desired smoothness. You enter the dimensions of the board. You choose how many layers to carve. Out comes `.svg` you can throw into CAD. Within Eisel, I made the following for my home state of Vermont.

![C06F2323-1F90-4C62-903F-07D728E7B05C_1_105_c](https://github.com/user-attachments/assets/7f0a4e68-f8b5-456d-b6b0-f8935fdc2826)

There are currently many on-going improvements to make. Among them:
- Making script compatible with larger states. I can download Vermont's and North Carolina's topography, but California is too big.
- Making script able to compose multiple boards. CNC machines generally work with 8 x 11 boards. I want to extend this software to make multiple blocks you can glue together
- Three dimensional rendering: I wish to use a software such as 3js to render what your design is from the application
- Application: I would love to make a website where you could draw any boundary, zoom into any mountain. From there, you should be able to make your own custom boundaries that can be more specific than simply finding boundaries state-by-state. 
