import bpy

# Path to your add-on's .zip file
addon_path = "D:/ProgramData/3D/BlenderAddOn/City_Generator_Addon_1.1.zip"
addon_name = "City_Generator_Addon"

# Check if the add-on is already installed
if addon_name not in bpy.context.preferences.addons:
    # Install the add-on if it's not already installed
    bpy.ops.preferences.addon_install(filepath=addon_path)
    print(f"{addon_name} installed.")
else:
    print(f"{addon_name} is already installed.")

# Enable the add-on
bpy.ops.preferences.addon_enable(module=addon_name)

# Save the preferences
bpy.ops.wm.save_userpref()

print(f"{addon_name} enabled and preferences saved successfully!")
