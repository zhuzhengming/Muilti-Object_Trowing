import os
import xml.etree.ElementTree as ET
import re

def process_xml_file(file_path, save_file_path, obj):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # (1) Replace "name" attribute with "banana"
        for elem in root.iter():
            if 'name' in elem.attrib:
                elem.attrib['name'] = obj

        # Find the line '<mesh file="textured.obj"/>'
        for elem in root.iter():
            if elem.tag == 'mesh' and 'file' in elem.attrib and elem.attrib['file'] == 'textured.obj':
                # Add 'name' attribute with value 'banana'
                elem.attrib['name'] = obj

        # (3) Add "name='banana_*'" after each "mesh" element with matching filenames
        for mesh_elem in root.iter('mesh'):
            file_attr = mesh_elem.attrib['file']
            match = re.search(r'textured_collision_(\d+)\.obj', file_attr)
            if match:
                number = match.group(1)
                mesh_elem.attrib['name'] = obj + '_' + number
                # new_elem = ET.Element(mesh_elem.tag, attrib=attrib)
                # root.insert(root.index(mesh_elem) + 1, new_elem)

            # (1) Replace values in "<geom material="material_0" mesh="textured" class="visual" />"
        for geom_elem in root.iter('geom'):
            if 'material' in geom_elem.attrib:
                geom_elem.attrib['material'] = obj
            if 'mesh' in geom_elem.attrib and geom_elem.attrib['mesh'] == 'textured':
                geom_elem.attrib['mesh'] = obj

            # (2) Replace values in all "<geom mesh="textured_collision_%d" class="collision" />"
        for geom_elem in root.iter('geom'):
            if 'mesh' in geom_elem.attrib:
                match = re.search(r'textured_collision_(\d+)', geom_elem.attrib['mesh'])
                if match:
                    number = match.group(1)
                    geom_elem.attrib['mesh'] = obj + "_" + number

            # (3) Replace values in "<material name="material_0" texture="texture_map" specular="0.5" shininess="0.5"/>"
        for material_elem in root.iter('material'):
            if 'texture' in material_elem.attrib and material_elem.attrib['texture'] == 'texture_map':
                material_elem.attrib['texture'] = obj
        # Write the updated XML content back to the file
        # tree.write(file_path, encoding='utf-8', xml_declaration=True)

        # Write the updated XML content to a new file
        tree.write(save_file_path, encoding='utf-8', xml_declaration=True)

        print(f"Processing of {file_path} completed successfully.")
    except FileNotFoundError:
        print("File not found.")
    except ET.ParseError:
        print("Invalid XML file.")


obj_name = ['banana', 'bottle', 'chip_can', 'soft_scrub', 'suger_box']
obj_name = ['soft_scrub']

for obj in obj_name[:2]:
    process_xml_file( obj + '/textured.xml', obj + '/' + obj + '.xml',  obj)

