# Introduction 
**Hi there!** Welcome to Fiona’s collection of climate models — a little space where I share models I’ve built out of curiosity and fun.
Some are inspired by existing tools and resources, while others are drafted from scratch based on core equations. A few may still be under construction or not fully functional (yet!), but I’ll keep updating things as I go.
Feel free to reach out if you’re interested or have questions — always happy to chat! :)

# Documentation 

## Energy Balance Models (EBMs)
Including two 0D EBMs and one 1D EMB. There is also a near-future projection inspired by University of Chicago's online course material.

Here’s what I got when forcing the 0D model with Holocene climate data (big thanks to Dr Matt Osman for suggesting the data sources):
![image](https://github.com/user-attachments/assets/24cbd15f-f7a4-45e6-bc3d-d969f95a0876)

And here’s the “40 kyr cycle” reproduced using orbital forcing in the 1D EBM. 
![image](https://github.com/user-attachments/assets/84e15761-8a2b-44a0-9851-7a8cdc50d7ad)



## Ice Flow Model 
This is a model I developed for my Part IB Glaciology coursework. I built it based on a base model provided by the department and added certain functions that I feel of interest. The final version incoroprates Glen flow law and Weertman sliding law to simulate glacier motion, and coupled a Degree Day Model to calculate mass balance chnage. Need glacier geometry and climate data to play around (I will upload the code with some sample input data after my coursework has got marked!). 

## "Bipolar Seesaw" Model
A ocean thermodynamics model I built for my Part IB Quaternary Environment supervision, following the fascinating paper by Stocker & Johnsen (2003). Thanks to Anna for helping refine the visualization! Here is a glance of its outputs...
![image](https://github.com/user-attachments/assets/f7aa0da5-eb85-4988-8f05-d9792b7c2ee2)
I also uploaded a more complicated version I built later which generates sawtooth-like Northern Hemisphere temperature, and also could be played with real temperature input (from NGRIP data).

## Atmosphere.ipynb: radiative-convective, atmospheric layer, and the orbital cycle
A notebook containing a series of atmospheric models, including radiative-convectice model, atmospheric layer model, the calculation of orbital cycle and the insolation.
Inspired by the Climlab tutorial provided by the University of Albany. 

## Ice Albedo and Elevation 
These are two models focusing on the response of ice sheets to the temperature and the feedback of this on planetary temperature. Inspired by the *Building Your Own Climate Model in Python* course provided by the University of Chicago on Couresa. 

## Shallow Water Model 
Try to simulate the change 

## Weather
This isn’t a full-fledged “climate model” — it’s a weather project I built during the Harvard CS50 course. It uses several APIs to fetch real-time weather data and also identifies the climate zone of the location. It might be especially fun for anyone curious about random geography facts! Feel free to explore — I personally find it super handy for checking travel destinations. Not just for planning outfits based on the weather, but also to get a better sense of the place’s geography :)
