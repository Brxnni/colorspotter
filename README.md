# The Plan

## Lines

* Find right contrast value such that multiple 1mm lines are found
  * Things that dont work: Harris Corner Detection
  * Found one: HoughLines P works well enough
* ~~Assign a thickness and angle to every line~~
* ~~Split lines into "thick" and "thin" ones~~
* Try to find lines that are parallel (i.e. have a similar angle)

## Circle

1. Find pixes of the right color (+- tolerance of course), starting from the center of the image

## Final

1. Find the lines closest to the purple circle -> two lines are 1mm apart
2. Use those to calculate the real-world area of the circle
