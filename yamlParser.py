#import json
import yaml

def yaml_parser(yaml_file):
    
    with open(yaml_file, 'r') as file: # convert yaml to json
        #data = yaml.safe_load(file)
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    classes_and_attributes = [] # list of class.attribute for every class and its attributes
    descriptions = {}
    
    # Classes which do not inherit from any other class
    root_classes = {
        class_name: details
        for class_name, details in data.get('classes', {}).items()
        if details.get('is_a') == "NamedEntity" or details.get('is_a') == "Triple" #or details.get('is_a') == "RelationshipType"
    }
    
    # adding classes descriptions
    for k in root_classes:
        if root_classes[k]["description"]:
            descriptions[k] = root_classes[k]["description"]
    
    # list of class.attribute from root classes (appended to classes_and_attributes)
    for class_name, details in root_classes.items():
        if "attributes" in details:
            for attribute in details["attributes"]:
                if attribute == "subject" or attribute == "object" or attribute == "predicate":
                    continue
                classes_and_attributes.append( class_name+"."+attribute )
                if "description" in details["attributes"][attribute]: # if the attribute has a description than save it
                    descriptions[class_name+"."+attribute] = details["attributes"][attribute]["description"]
            
    # Classes which inherit from one or more classes
    hereditary_classes = {
        class_name: details
        for class_name, details in data.get('classes', {}).items()
        if details.get('is_a') != "NamedEntity" and details.get('is_a') != "Triple" and details.get('is_a') != "RelationshipType"
    }
    
    # adding classes descriptions
    for k in hereditary_classes:
        if hereditary_classes[k]["description"]:
            descriptions[k] = hereditary_classes[k]["description"]
    
    inherited_attributes = [] # list of inherited attributes of the curent class
    for class_name, details in hereditary_classes.items(): # loop over every inherited class
        
        if "attributes" in details:
            for attribute in details["attributes"]: # loop over attributes of the current class
                if attribute == "subject" or attribute == "object" or attribute == "predicate":
                    continue
                inherited_attributes.append(attribute)
                if "description" in details["attributes"][attribute]: # if the attribute has a description than save it
                    descriptions[class_name+"."+attribute] = details["attributes"][attribute]["description"]
        is_a = details.get('is_a') # the current class
            
        # loop over the inheriting hierarchy of the current class
        while is_a and is_a != "NamedEntity" and is_a != "RelationshipType" and is_a != "Triple":
            try:
                if is_a in root_classes:
                    inherited_class = root_classes[is_a]
                    is_a = root_classes[is_a].get('is_a')
                elif is_a in hereditary_classes:
                    inherited_class = hereditary_classes[is_a]
                    is_a = hereditary_classes[is_a].get('is_a')
                else:
                    # If is_a is not found in either dictionary, break the loop
                    print(f"Warning: Class '{is_a}' not found in class definitions")
                    break
                
                # is_a now is the next class in the inheriting hierarchy of the current analyzed class (outer loop)
                
                # update the attributes of the current class with the attributes it inherits from its inheriting hierarchy
                if "attributes" in inherited_class:
                    for attribute in inherited_class["attributes"]:
                        if attribute == "subject" or attribute == "object" or attribute == "predicate":
                            continue
                        inherited_attributes.append(attribute)
                        if "description" in inherited_class["attributes"][attribute]: # if the attribute has a description than save it
                            descriptions[class_name+"."+attribute] = inherited_class["attributes"][attribute]["description"]
            except Exception as e:
                print(f"Error processing inheritance for class '{class_name}': {str(e)}")
                break
            
        for attribute in inherited_attributes:
            classes_and_attributes.append(class_name+"."+attribute) # update the class.attribute list
                
        inherited_attributes = [] # reset the list of attributes for the next inherited class in the loop
    
    print(f"loaded yaml file with n. {len(classes_and_attributes)} classes and {len(descriptions)} descriptions")
    return classes_and_attributes, descriptions

# classes_and_attributes, descriptions = yaml_parser(f"./yamls/PKT.yaml")

# print("classes_and_attributes: ", classes_and_attributes)
# print("*"*100)
# print("descriptions: ", descriptions)