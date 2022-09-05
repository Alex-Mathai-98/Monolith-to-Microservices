""" Several utilities used by other classes in performing partition validation """

class Node(object):
    """
    Represents a node in the graph
    """

    def __init__(self, nodename, classname, methodname):
        self.nodename = nodename
        self.classname = classname
        self.methodname = methodname

        self.annotations = set()
        self.outgoing_links = {}

    def get_outgoing_links(self):
        return self.outgoing_links

    def add_annotation(self, annotation):
        self.annotations.add(annotation)

    def get_name(self):
        return self.nodename

    def get_classname(self):
        return self.classname

    def get_annotations(self):
        return self.annotations

    def get_outgoing_links(self):
        return self.outgoing_links

    def add_outgoing_link(self, node):
        self.outgoing_links[node.get_name()] = node

class ClassNode(object):
    """
    Represents a class node in the graph
    """

    def __init__(self, nodename):
        self.nodename = nodename

        self.annotations = set()
        self.outgoing_links = {}

    def get_outgoing_links(self):
        return self.outgoing_links

    def add_annotation(self, annotation):
        self.annotations.add(annotation)

    def get_name(self):
        return self.nodename

    def get_annotations(self):
        return self.annotations

    def get_outgoing_links(self):
        return self.outgoing_links

    def add_outgoing_link(self, node):
        self.outgoing_links[node.get_name()] = node


def parse_callgraph_line(line):
    """
    Parses a line from the call graph file
    """
    parts = line.split("->")
    if len(parts) != 2:
        #print("Bad line:", line)
        return None, None

    return process_line_part(parts[0]), process_line_part(parts[1])

def process_line_part(linepart):
    """
    Processes individual node info from a line part
    """
    linepart = linepart.strip().replace("\"", "")

    parts = linepart.split("] ")
    if len(parts) > 2:
        return None
    annotation = None
    if len(parts) == 2:
        annotation = parts[0].replace("]", "")
        annotation = annotation.replace("[", "")

    nodename = parts[-1]
    nodepath = nodename.split(".")
    methodname = nodepath[-1]
    classname = nodename
    if len(nodepath) > 1:
        classname = ".".join(nodepath[:-1])
    

    node =  Node(nodename, classname, methodname)
    if annotation is not None:
        annotation = clean_annotation(annotation)
        node.add_annotation(annotation)

    return node

def clean_annotation(annotation):
    tx_index = annotation.find(", txid: ")
    if tx_index > -1:
        annotation = annotation[:tx_index]+"}"
        lastfewchars = annotation[-10:]
        paren_idx = lastfewchars.rfind("(")
        if paren_idx > -1:
            paren_idx = 10 - paren_idx
            
            annotation = annotation[:-paren_idx] + "}"
            
    return annotation