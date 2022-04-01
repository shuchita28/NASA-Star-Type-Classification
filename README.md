# 🌠: NASA-Star-Type-Classification

## Project
To predict the Type of star based on data such as Temperature of surface, luminosity of star, color and other physical attributes.

## Data
https://www.kaggle.com/datasets/brsdincer/star-type-classification

## Variable definition
### Predictor variables 
Temperature : Average temperature of the surface of star measured in degree K
L : Realtive Luminosity of the star with respect to the sun measured in L/Lo units
R : Relative radius of the star with respect to the sun measured in R/Ro units
AM: Absolute Mass of the star
Color : General Color of Spectrum
Spectral_Class : O,B,A,F,G,K,M / SMASS - https://en.wikipedia.org/wiki/Asteroid_spectral_types

### Target variable 
Type : Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , Super Giants, Hyper Giants

One hot encoded from categorical to numeric (0 to 5) as follows :

 - Red Dwarf - 0
 - Brown Dwarf - 1
 - White Dwarf - 2
 - Main Sequence - 3
 - Super Giants - 4
 - Hyper Giants - 5

## Math and formulae :

Lo = 3.828 x 10^26 Watts (Avg Luminosity of Sun)
Ro = 6.9551 x 10^8 m (Avg Radius of Sun)

## Deployment
Using streamlit 

