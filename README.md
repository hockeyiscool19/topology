# Introduction

This repository automates making design for CNC machines. That is, you choose a desired state to make a map. You choose a desired smoothness. You enter the dimensions of the board. You choose how many layers to carve. Out comes `.svg` you can throw into CAD. Within Eisel, I made the following for my home state of Vermont. From Eisel, as with other CADing software, you are able to create `g-code` -- i.e., directions for what the CNC machine should carve away.

![C06F2323-1F90-4C62-903F-07D728E7B05C_1_105_c](https://github.com/user-attachments/assets/7f0a4e68-f8b5-456d-b6b0-f8935fdc2826)

In essence, the script works in the following way:
1) Provide inputs -- smoothing factor, state/abbreviation, number of layers, etc...
2) Script download and caches the `tif` files
3) Finds minimum and maximum elevation.
4) Based on number of layers, min and max elevation, the script creates bands of elevation.
5) Based on the downloaded map, the script makes pictures representing the different bands of elevation.
  - For example, if the minimimum height is 0ft and max heigh is 10,000 feet, and you choose 5 layers -- then you will have 5 resulting images. Each represent the land at that particular height. I.E., the first picture will represent land from 0 - 1,999, picture 2 will represent land 2,000 - 3,999ft in elevation, etc.
6) Copy and paste pictures into CADing software... This could be updated in the future with more development time :)

Note: if you think about it, land can we quite jagged. I apply a "smoothing factor" so that the drill bit will not shave off every part of the wood. Additionally, if an area has dramatic elevation differences, the smoothing factor rounds up for areas with high elevation, and it rounds down for areas with low elevation. This creates a more dramatic mountainess effect in the final product. 

There are currently many on-going improvements to make. Among them:
- Making script compatible with larger states. I can download Vermont's and North Carolina's topography, but California is too big.
- Making script able to compose multiple boards. CNC machines generally work with 8 x 11 boards. I want to extend this software to make multiple blocks you can glue together
- Three dimensional rendering: I wish to use a software such as 3js to render what your design is from the application
- Application: I would love to make a website where you could draw any boundary, zoom into any mountain. From there, you should be able to make your own custom boundaries that can be more specific than simply finding boundaries state-by-state. 
