# Introduction 
**Hi there!** Welcome to Fiona’s collection of climate models — a little space where I share models I’ve built out of curiosity and fun.
Some are inspired by existing tools and resources, while others are drafted from scratch based on core equations. A few may still be under construction or not fully functional (yet!), but I’ll keep updating things as I go.
Please feel free to reach out if you’re interested or have questions, I am always happy to chat! :)

# Documentation 

## Energy Balance Models 
Including two 0D EBMs and one 1D EMB. There is also a near-future projection inspired by University of Chicago's online course material.

Here’s what I got when forcing the 0D model with Holocene climate data (big thanks to Dr Matt Osman for suggesting the data sources):
![image](https://github.com/user-attachments/assets/24cbd15f-f7a4-45e6-bc3d-d969f95a0876)

And here’s the “40 kyr cycle” reproduced using orbital forcing in the 1D EBM. 
![image](https://github.com/user-attachments/assets/84e15761-8a2b-44a0-9851-7a8cdc50d7ad)

(I am also trying to compare it with Maximum Entropy Production model developed by Lorenz et al 2001, which I find really intriguing...)

## "Bipolar Seesaw" Model
A ocean thermodynamics model I built for my Part IB Quaternary Environment supervision, following the fascinating paper by Stocker & Johnsen (2003). Thanks to Anna for helping refine the visualization! Here is a glance of its outputs...
![image](https://github.com/user-attachments/assets/f7aa0da5-eb85-4988-8f05-d9792b7c2ee2)
I also uploaded a more complicated version I built later which generates sawtooth-like Northern Hemisphere temperature, and also could be played with real temperature input (from NGRIP data).

## Shallow Water Model 
Here, I am solving a linear rotating shallow water system with gravity waves. I set up a tiny grid, put a small "tower" of extra fluid height in the centre, and let gravity and the Coriolis force evolve it forward. The equations include:

\[
\frac{\partial u}{\partial t} = fv - g\,\frac{\partial \eta}{\partial x} - r u + \tau_x,
\qquad
\frac{\partial v}{\partial t} = -fu - g\,\frac{\partial \eta}{\partial y} - r v,
\qquad
\frac{\partial \eta}{\partial t} = -H_0\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right).
\]

Where \(u, v\) are velocity components; 
\(\eta\) is the free-surface height anomaly; 
\(f\) is Coriolis parameter; 
\(g\) is gravity (I use a reduced value so the model evolves slowly); 
\(H_0\) is the background depth; 
\(r\) is linear drag; 
and \(\tau_x\) is optional wind forcing. 

By running the model, you should first see the bump radiating gravity waves, which then getting bent by rotation and eventually settling into a geostrophic swirl where pressure gradients and Coriolis forces balance.

## Ice Albedo and Elevation 
These are two models focusing on the response of ice sheets to the temperature and the feedback of this on planetary temperature. Inspired by the *Building Your Own Climate Model in Python* course provided by the University of Chicago on Coursera.

## Glacier Mass Balance and Flow  
This is a model I developed for my Part IB Glaciology coursework. I built it based on a base model provided by the department and added certain functions that I feel of interest. The final version incoroprates Glen flow law and Weertman sliding law to simulate glacier motion, and coupled a Degree Day Model to calculate mass balance chnage. (I will upload the core part of the code after my coursework has got marked!)

## Atmosphere
This is a notebook containing a series of atmospheric models, including radiative-convectice model, atmospheric layer model, the calculation of orbital cycle and the insolation.
The script development is inspired by the *Climlab tutorial* provided by the University of Albany. 

## Weather
This isn’t a full-fledged “climate model” — it’s a weather project I built during the *Harvard CS50* course. It uses several APIs to fetch real-time weather data and also identifies the climate zone of the location. It might be especially fun for anyone curious about random geography facts! Feel free to explore — I personally find it super handy for checking travel destinations. Not just for planning outfits based on the weather, but also to get a better sense of the place’s geography :) 

## Others 
* Joint UK Land Surface Simulator (JULES) python model
* High Altitude Balloon 
