import os
import numpy as np
import pickle
import data_layer.utilities.generalutils as utils

class CallGraph(object):
    """
    Represents a call graph at both method level and class level
    Contains structures to hold graph information as well as
    additional information about the nodes
    """

    def __init__(self, callgraph_file, filtered_classes, entrypoints, build_cooc_matrix=False):
        self.allnodes = {}
        self.allclassnodes = {}
        self.filtered_classes = filtered_classes
        self.entrypoints = entrypoints

        self.build_callgraph(callgraph_file, entrypoints)
        self.build_class_callgraph()
        if build_cooc_matrix:
            self.build_class_cooccurence_matrix(True)

    def build_callgraph(self, callgraph_file, entrypoints):
        with open(callgraph_file, 'r', encoding="utf-8") as f:
            count = 0
            for line in f:
                if not line[0] == "\"":
                    continue
                nodefrom, nodeto = utils.parse_callgraph_line(line.strip())
                # Check if we got valid nodes from parsing
                if nodefrom is None or nodeto is None:
                    #print("Invalid node returned from parsing. Skipping")
                    continue

                # If we have not seen these nodes before, add them to node list
                # If we have seen them, make sure their new annotations are saved
                for node in (nodefrom, nodeto):
                    existingnode = self.allnodes.get(node.get_name())
                    if existingnode is None:
                        self.allnodes[node.get_name()] = node
                    else:
                        for anno in node.get_annotations():
                            existingnode.add_annotation(anno)

                # Get the saved versions of the nodes
                nodefrom = self.allnodes[nodefrom.get_name()]
                nodeto = self.allnodes[nodeto.get_name()]

                # Make the link
                nodefrom.add_outgoing_link(nodeto)
                count += 1
                if count % 1000000 == 0:
                    #print("Parsed", count, "lines so far ..")
                    pass

            #print("Parsed", count, "lines from the file:", callgraph_file)
            #print("The graph has", len(self.allnodes), "nodes.")

    def build_class_callgraph(self):
        """
        Takes the method level call graph and generates a class level call graph
        """
        # Create entries in the list for each node
        for nodename in self.allnodes:
            node = self.allnodes[nodename]
            classname = node.get_classname()

            if classname not in self.filtered_classes:
                continue

            classnode = self.allclassnodes.get(classname)
            if classnode is None:
                classnode = utils.ClassNode(classname)
                self.allclassnodes[classname] = classnode

        # Now add links and annotations since all possible nodes have been saved
        for nodename in self.allnodes:
            node = self.allnodes[nodename]
            if node.get_classname() not in self.filtered_classes:
                continue

            classnode = self.allclassnodes[node.get_classname()]

            outlinks = node.get_outgoing_links()
            for outnodename in outlinks.keys():
                outnode = outlinks[outnodename]
                outclassname = outnode.get_classname()
                if outclassname not in self.filtered_classes:
                    continue
                if node.get_classname() == outclassname:
                    continue
                outclassnode = self.allclassnodes[outclassname]

                classnode.add_outgoing_link(outclassnode)

            for anno in node.get_annotations():
                classnode.add_annotation(anno)

        #print("The class graph has", len(self.allclassnodes), "nodes.")

        # compute class and entrypoint association
        self.class2entrypoint = {}
        self.entrypoint2class = {}

        for classnode in self.allclassnodes.values():
            ep_arr = list(classnode.get_annotations())
            classname = classnode.get_name()
            self.class2entrypoint[classname] = ep_arr
            for ep_name in ep_arr:
                classes = self.entrypoint2class.get(ep_name)
                if classes is None:
                    classes = []
                classes.append(classname)
                self.entrypoint2class[ep_name] = classes


    def get_outgoing_classnodes(self, classname):
        classnode = self.allclassnodes[classname]
        return classnode.get_outgoing_links()

    def build_class_cooccurence_matrix(self, bidirectional=False):
        # init counts and dicts
        self.class_forward_counts = {}
        self.class_base_counts = {}
        self.visited = {}
        self.uniquepath_count = 0
        self.allclassesfrommethods = []

        #print("Generating base counts ..")

        for classname in self.filtered_classes:
            self.class_forward_counts[classname] = {}
            self.class_base_counts[classname] = 0
            for secondclass in self.filtered_classes:
                self.class_forward_counts[classname][secondclass] = 0

        #print("Populating actual counts ..")

        for entrypoint_name, methods in self.entrypoints.items():
            for ep_methodname in methods:
                
                if self.allnodes.get(ep_methodname) is not None:
                    current_class = self.allnodes[ep_methodname].get_classname()
                    if current_class in self.filtered_classes:
                        pathsofar = [ep_methodname]
                        self.dfs_traverse_new(ep_methodname, entrypoint_name, pathsofar, bidirectional)

        #print("Total unique paths:", self.uniquepath_count)

    def dfs_traverse_new(self, methodname, entrypoint_name, pathsofar, bidirectional):
        currentnode = self.allnodes[methodname]
        neighbors = currentnode.get_outgoing_links().keys()
        unvisited_nodes = []
        pathsofar = [methodname]

        for n in neighbors:
            neighbor_node = self.allnodes[n]
            neighborclass = neighbor_node.get_classname()

            if entrypoint_name in neighbor_node.get_annotations() and neighborclass in self.filtered_classes:
                unvisited_nodes.append(neighbor_node)
                pathsofar.append(n)

        while len(unvisited_nodes) > 0:
            nextnode = unvisited_nodes.pop()
            if nextnode is None:
                pathsofar.pop()
                continue
            # pathsofar.append(nextnode)

            neighbors = nextnode.get_outgoing_links().keys()
            valid_count = 0

            unvisited_nodes.append(None)
            for n in neighbors:
                neighbor_node = self.allnodes[n]
                neighborclass = neighbor_node.get_classname()
                if entrypoint_name in neighbor_node.get_annotations() and neighborclass in self.filtered_classes:
                    if n not in pathsofar:
                        unvisited_nodes.append(neighbor_node)
                        pathsofar.append(n)
                        valid_count += 1

            if valid_count == 0:
                self.process_path(pathsofar, bidirectional)
            if len(pathsofar) > 20:
                continue

    def process_path(self, path, bidirectional):
        self.uniquepath_count += 1
        pathclassnames = []
        for pathnode in path:
            classname = self.allnodes[pathnode].get_classname()
            if classname not in pathclassnames: # new class name not seen so far in the path
                pathclassnames.append(classname)
            else:
                if classname != pathclassnames[-1]: # not a new class but seen at least 2 hops ago
                    pathclassnames.append(classname)

        for classname in pathclassnames:
            self.class_base_counts[classname] += 1

        maxidx = len(pathclassnames)
        if bidirectional:
            for i in range(maxidx):
                for j in range(maxidx):
                    firstclass = pathclassnames[i]
                    secondclass = pathclassnames[j]
                    countdict = self.class_forward_counts[firstclass]
                    countdict[secondclass] += 1
                    self.class_forward_counts[firstclass] = countdict
        else:
            for i in range(maxidx):
                for j in range(i,maxidx):
                    firstclass = pathclassnames[i]
                    secondclass = pathclassnames[j]
                    countdict = self.class_forward_counts[firstclass]
                    countdict[secondclass] += 1
                    self.class_forward_counts[firstclass] = countdict

    # def dfs_traverse_new(self, methodname, entrypoint_name, pathsofar, bidirectional):

    #     currentnode = self.allnodes[methodname]
    #     visited_nodes = {methodname:True}
    #     neighbors = list(currentnode.get_outgoing_links().keys())
    #     neighbors.sort()
    #     unvisited_nodes = []
    #     basepath = (currentnode, [methodname])
    #     max_limit = 10000

    #     for n in neighbors:
    #         neighbor_node = self.allnodes[n]
    #         neighborclass = neighbor_node.get_classname()

    #         if entrypoint_name in neighbor_node.get_annotations() and neighborclass in self.filtered_classes:
    #             if len(visited_nodes) > max_limit :
    #                 # limit reached
    #                 break
    #             if visited_nodes.get(n) == True :
    #                 # already visited this neighbour
    #                 continue
    #             else :
    #                 # visiting a new neighbour
    #                 visited_nodes[n] = True
    #                 unvisited_nodes.append((neighbor_node, basepath[1]+[n]))

    #     while len(unvisited_nodes) > 0 :

    #         if len(unvisited_nodes) > max_limit :
    #             assert(False)

    #         (nextnode, pathsofar) = unvisited_nodes[-1]
    #         unvisited_nodes = unvisited_nodes[:-1]
    #         neighbors = list(nextnode.get_outgoing_links().keys())
    #         neighbors.sort()
    #         valid_count = 0
    #         for n in neighbors:
    #             neighbor_node = self.allnodes[n]
    #             neighborclass = neighbor_node.get_classname()
    #             if entrypoint_name in neighbor_node.get_annotations() and neighborclass in self.filtered_classes:
    #                 if visited_nodes.get(n,None) == None :
    #                     if len(visited_nodes) > max_limit :
    #                         continue
    #                     else :
    #                         visited_nodes[n] = True
    #                         unvisited_nodes.append((neighbor_node, pathsofar+[n]))
    #                         valid_count += 1

    #         if valid_count == 0:
    #             self.process_path(pathsofar, bidirectional)
    #         if len(pathsofar) > 20:
    #             continue

    # def process_path(self, path, bidirectional):
    #     pathclassnames = []
    #     for methodname in path:
    #         classname = self.allnodes[methodname].get_classname()
    #         if classname not in pathclassnames: # new class name not seen so far in the path
    #             pathclassnames.append(classname)
    #         else:
    #             if classname != pathclassnames[-1]: # not a new class but seen at least 2 hops ago
    #                 pathclassnames.append(classname)

    #     for classname in pathclassnames:
    #         self.class_base_counts[classname] += 1

    #     maxidx = len(pathclassnames)
    #     if bidirectional:
    #         for i in range(maxidx):
    #             for j in range(maxidx):
    #                 firstclass = pathclassnames[i]
    #                 secondclass = pathclassnames[j]
    #                 countdict = self.class_forward_counts[firstclass]
    #                 countdict[secondclass] += 1
    #                 self.class_forward_counts[firstclass] = countdict
    #     else:
    #         for i in range(maxidx):
    #             for j in range(i,maxidx):
    #                 firstclass = pathclassnames[i]
    #                 secondclass = pathclassnames[j]
    #                 countdict = self.class_forward_counts[firstclass]
    #                 countdict[secondclass] += 1
    #                 self.class_forward_counts[firstclass] = countdict

    def construct_the_matrix(self, class2idx, class2resource, normalize=False):
        """
        self.allclassnodes and self.class_forward_counts has only some nodes
        construct the matrix in terms of all filtered classes
        """

        # Before this point the forward counts values are populated for call graph classes only
        # We also need to add counts for class-resource interactions
        # So we compute them now before generating the normalized matrix
        # Also, we use the class's base count value for the resource interaction
        # This would imply everytime a class is accessed, all its resources are accesed
        for classname in class2resource.keys():
            if class2idx.get(classname,-1) == -1 :
                # skipping programs that have only CRUD edges and no program to program interactions
                continue
            classidx = class2idx[classname]
            basecount = self.class_base_counts[classname]
            for resourcename in class2resource[classname]:
                #resourceidx = class2idx[resourcename]
                self.class_forward_counts[classname][resourcename] = basecount
                self.class_forward_counts[resourcename][classname] = basecount
                if self.class_base_counts.get(resourcename) is None:
                    self.class_base_counts[resourcename] = 0
                self.class_base_counts[resourcename] += basecount

        num_classes = len(self.filtered_classes)
        matrix = np.zeros((num_classes, num_classes))

        for row in self.class_forward_counts.keys():
            r_idx = class2idx[row]
            for col in self.class_forward_counts[row].keys():
                value = self.class_forward_counts[row][col]
                c_idx = class2idx[col]
                if normalize:
                    if self.class_base_counts[row] > 0:
                        matrix[r_idx,c_idx] = value / float(self.class_base_counts[row])
                else:
                    matrix[r_idx,c_idx] = value

        self.cooccurence_matrix = matrix
        return matrix

    def release_memory(self):
        """Release memory objects after the primary data structures have been populated"""
        if self.build_class_cooccurence_matrix:
            del self.class_forward_counts
            del self.class_base_counts
        del self.visited
        #del self.class2entrypoint
        #del self.entrypoint2class
        self.allnodes = {}
