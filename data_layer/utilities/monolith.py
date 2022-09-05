import os
import json
import numpy as np
from data_layer.utilities.graph import CallGraph 

class Monolith():
    """
    Represents the monolith application. Stores and processes all relevant static analysis results
    Static analysis is not done here and is expected to be performed elsewhere
    """
    def __init__(self, icu_fie, callgraph_file, entrypoint_file, resource_file, build_cooc_matrix=False):

        self.icu = []
        self.implements = []
        self.filtered_classes = []
        self.num_icu_classes = 0
        self.resources = []

        self.icuclass2idx = {}
        self.idx2icuclass = {}

        self.resource2class = {}
        self.class2resource = {}
        self.resource_usage_map = {}

        self.parse_resources(resource_file)
        #print("parsing ICU ..")
        self.parse_inter_class_usage(icu_fie)

        if callgraph_file is not None and os.path.exists(callgraph_file):
            #print("parsing entrypoints ..")
            self.entrypoints = self.parse_entrypoints(entrypoint_file)

            #print("parsing call-graph ..")
            self.graph = CallGraph(callgraph_file, self.filtered_classes, self.entrypoints, build_cooc_matrix=build_cooc_matrix)
            if build_cooc_matrix:
                #print("building matrix")
                self.graph.construct_the_matrix(self.icuclass2idx, self.class2resource, True)
                self.graph.release_memory()

    def parse_entrypoints(self, entrypoint_file):
        with open(entrypoint_file, 'r') as f:
            data = json.load(f)

        entrypoints = {}
        for item in data:
            name = item["service_entry_name"]
            name = name.replace("[", "")
            name = name.replace("]", "")
            classes = item["class_method_name"]
            entrypoints[name] = classes

        return entrypoints
    
    def parse_resources(self, resource_file):
        if not os.path.exists(resource_file):
            #print("Resources file not available.")
            return

        with open(resource_file, "r") as f:
            data = json.load(f)
        self.resources = []

        #print("Parsing resources ...")
        resource_idx = 0
        for entry in data:
            service_name = entry["service_name"]
            resource_type = "res"
            if entry.get("resource_type") is not None:
                resource_type = entry["resource_type"]
            if entry.get("resource_name") is not None:
                resource_name = entry["resource_name"] 
            if entry.get("db_name") is not None:
                resource_name = entry["db_name"] 
            if entry.get("resource") is not None:
                resource_name = entry["resource"] 
            usage_type = "CRUD_" + entry["crud"]

            # add type info to the resource name
            #resource_name = "-*-*-"+resource_name+"-"+resource_type
            resource_name = resource_name+"-"+resource_type
            resource_name = resource_name.lower()  # ignore case in the names
            parts = service_name.split(".")
            if len(parts) == 1:
                classname = service_name
            else:
                classname = ".".join(parts[:-1])
            
            if self.resource_usage_map.get((resource_name, classname)) is None:
                self.resource_usage_map[(resource_name, classname)] = []
            self.resource_usage_map[(resource_name, classname)].append(usage_type)

            if self.resource2class.get(resource_name) is None:
                self.resource2class[resource_name] = [classname]
                # if this is a new resource, also add entry to ICU
                #newidx = resource_idx + icu_idx_offset
                if self.icuclass2idx.get(resource_name) is None:
                    self.icuclass2idx[resource_name] = resource_idx
                    self.idx2icuclass[resource_idx] = resource_name
                    resource_idx += 1
                    self.resources.append(resource_name)
            else:
                self.resource2class[resource_name].append(classname)

            if self.class2resource.get(classname) is None:
                self.class2resource[classname] = [resource_name]
            else:
                self.class2resource[classname].append(resource_name)

    def parse_inter_class_usage(self, icu_file):
        """
        Parse the interclass usage file. Generates multiple data structures.
        """
        #print("Parsing interclass usage ..")
        with open(icu_file, "r") as fp:
            icu_json = json.load(fp)

        # since we already have resources in the ICU, we offset the classes idxs
        idx_offset = len(self.resources)

        self.original_icu_classes = list(icu_json.keys())
        self.num_icu_classes = len(self.original_icu_classes)
        #print("Found", self.num_icu_classes, "classes in inter class usage file.")

        for i,classname in enumerate(self.original_icu_classes):
            self.icuclass2idx[classname] = i + idx_offset
            self.idx2icuclass[i + idx_offset] = classname

        self.all_icu_classes = self.resources + self.original_icu_classes
        # recoompute count, with resources now included
        self.num_icu_classes = len(self.all_icu_classes)

        # set them as filtered classes, will be updated later if we use filters
        self.filtered_classes = self.original_icu_classes
        # set them as filtered classes, will be updated later if we use filters
        self.filtered_classes = self.all_icu_classes

        for i,classname in enumerate(self.all_icu_classes):
            self.icuclass2idx[classname] = i
            self.idx2icuclass[i] = classname

        self.icu = np.zeros((self.num_icu_classes, self.num_icu_classes), dtype=np.int)
        # the icu now actually contains both classes and resources
        # but inheritance is only between classes, so some part of this matrix is actually unnecessary
        # but separating it would lead to very complex code, so its left as it is
        self.implements = np.zeros((self.num_icu_classes, self.num_icu_classes), dtype=np.int)

        # Populate the ICU matrix with class-class information
        for classname in self.all_icu_classes:
            # If we encounter a resource, update class-resource info
            if icu_json.get(classname) is None:
                residx = self.icuclass2idx[classname]
                classes = self.resource2class[classname]
                for conn_classname in classes:
                    # check if class is in ICU
                    if self.icuclass2idx.get(conn_classname,-1) != -1 :
                        classidx = self.icuclass2idx[conn_classname]
                        self.icu[residx][classidx] += 1
                        self.icu[classidx][residx] += 1
                        
                    else :
                        # if program is in resource.json and not in inter_class_usage.json then ignore the program
                        pass
                # move on since we wont have any info in the icu json
                continue

            # if we reach here it means we are processing an icu class
            # so look in the icu json and extract details
            classdata = icu_json[classname]

            used_class_dict = classdata["usedClassesToCount"]
            usedby_class_dict = classdata["usedByClassesToCount"]

            # Outgoing edges - this links classes to other classes
            row = self.icuclass2idx[classname]
            for k,v in used_class_dict.items():
                col = self.icuclass2idx[k]
                self.icu[row][col] = v

            # Incoming edges - this links classes to other classes
            col = self.icuclass2idx[classname]
            for k,v in usedby_class_dict.items():
                row = self.icuclass2idx[k]
                self.icu[row][col] = v

            # Parent classes and interfaces - links classes to classes
            c_idx = self.icuclass2idx[classname]
            parent = classdata.get("superClass")
            interfaces = classdata.get("implementedInterfaces")
            if parent is not None and parent[:4] != "java" and len(parent) > 3:
                if self.icuclass2idx.get(parent) is not None:
                    parentidx = self.icuclass2idx[parent]
                    self.implements[c_idx][parentidx] = 1
                    self.implements[parentidx][c_idx] = 1
            if interfaces is not None:
                for interface_name in interfaces:
                    if interface_name[:4] != "java" and self.icuclass2idx.get(interface_name) is not None:
                        interface_idx = self.icuclass2idx[interface_name]
                        self.implements[c_idx][interface_idx] = 1
                        self.implements[interface_idx][c_idx] = 1

    def get_classes_used_by(self, classname):
        icu_idx = self.icuclass2idx[classname]
        icu_row = self.icu[icu_idx]

        selected_classes = []
        for idx, count in enumerate(icu_row):
            if count > 0:
                classname = self.idx2icuclass[idx]
                if classname in self.filtered_classes:
                    selected_classes.append(classname)
        return selected_classes

    def get_classes_using(self, classname):
        icu_idx = self.icuclass2idx[classname]
        icu_col = self.icu[:,icu_idx]

        selected_classes = []
        for idx, count in enumerate(icu_col):
            if count > 0:
                classname = self.idx2icuclass[idx]
                if classname in self.filtered_classes:
                    selected_classes.append(classname)
        return selected_classes
    
    def get_all_edges(self) :
        ans_dict = {}
        for class_ in self.filtered_classes :
            if "-" in class_: # skip resources in this step
                continue

            in_edges = self.get_classes_using(class_)
            out_edges = self.get_classes_used_by(class_)
            in_edges = [e for e in in_edges if "-" not in e]    # skip resources
            out_edges = [e for e in out_edges if "-" not in e]  # skip resources
            ans_dict[class_] = {}
            ans_dict[class_]["in"] = in_edges
            ans_dict[class_]["in_types"] = ["CALLS" for _ in in_edges]
            ans_dict[class_]["out"] = out_edges
            ans_dict[class_]["out_types"] = ["CALLS" for _ in out_edges]
        
        # include resource edges (this could be included in the earlier if conditions as well)
        # kept it here for more customizability
        for resource in self.resource2class.keys():
            resource_key = "resource_#_"+resource
            ans_dict[resource_key] = {}
            ans_dict[resource_key]["in"] = []
            ans_dict[resource_key]["in_types"] = []
            for classname in self.resource2class[resource]:
                usage_types = self.resource_usage_map[(resource, classname)]
                for u_t in usage_types:
                    if ans_dict.get(classname,-1) != -1 :
                        ans_dict[classname]["out"].append(resource_key)
                        ans_dict[classname]["out_types"].append(u_t)
                        ans_dict[resource_key]["in"].append(classname)
                        ans_dict[resource_key]["in_types"].append(u_t)
                

            ans_dict[resource_key]["out"] = []  # resources have no outgoing edges
            ans_dict[resource_key]["out_types"] = []  # resources have no outgoing edge types

        return ans_dict