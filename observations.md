# MultiPoint

1. Adding L1-like penalty improved results HUUUGE, like from 10cm accuracy up to 1-2cm. Before activity map was showing high activity in voxels close to detector, also close to true source, and it was showing activity in many different places
    Also places where activity was, were quite big

Before (without L1-like penalty): \
![alt text](image.png) \
After (with L1-like penalty): \
![alt text](image-1.png)
    
2. L1-like penalty does not work for multi sources, general idea how to handle more then 1 source: run ml-em to find first source, then subtract it from measurements and run ml-em again to find second source, repeat until no more sources are found. THIS IS GENERAL IDEA, NOT HOW EXACTLY SHOULD WORK.
