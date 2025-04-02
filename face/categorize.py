import os

# Replace "directory_path" with the path of your directory
directory_path = "//videos/train/positive"

# Replace "old_extension" with the current extension of your video files
old_extension = ".mp4.PFILE", ".MOV.PFILE"

# Replace "new_extension" with the new extension you want to give to your video files
new_extension = ".mp4"

# Replace "name_pattern" with the pattern you want to use for the new names
# Use curly braces {} to represent the position where the number should be inserted
# For example, "video_{}.mp4" will generate names like "video_1.mp4", "video_2.mp4", etc.
name_pattern = "video_{}.mp4"

# Initialize a counter to keep track of the number to insert in the name pattern
counter = 1

# Loop through all the files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a video file with the current extension
                    if filename.endswith(old_extension):
                        # Create a new name with the specified pattern and the current counter value
                        new_name = name_pattern.format(counter)
                        # Increment the counter for the next file
                        counter += 1
                        # Rename the file with the new name and extension
                        os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_name))
