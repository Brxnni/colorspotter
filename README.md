# The Plan

## Lines

1. Find right contrast value such that enough lines are found for the following steps to make sense
    - Things that dont work: Harris Corner Detection
2. Assign a thickness and angle every line
3. Split lines into "thick" and "thin" ones
4. Try to find thick lines that are parallel (i.e. have a similar angle)

## Circle

1. Find pixes of the right color (+- tolerance of course), starting from the center of the image

## Final

1. Find the lines closest to the purple circle -> two thick lines are 1cm apart
2. Use those to calculate the real-world area of the circle
