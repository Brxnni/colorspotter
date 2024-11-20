# The Plan

## Lines

* ✅ Find right contrast value such that multiple 1cm lines are found
* ✅ Try to find lines that are parallel (i.e. have a similar angle)

## Circle

* ✅ Find pixes of the right color (+- tolerance of course), starting from the center of the image
* Fill out blob shape from within such that pixels inside the blob are counted too, even if they don't fit the criteria

## Final

* Find the lines closest to the purple circle -> two lines are 1mm apart
* Use those lines' pixel distances to calculate the relationship between pixels and real-world units -> calculate actual area of the drop in the image
