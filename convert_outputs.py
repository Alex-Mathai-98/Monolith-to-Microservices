import sys
import json
import os
import time
from data_layer.utilities.monolith import Monolith

class OutputFormatter(object):

    def __init__(self, clusters, icu_file, callgraph_file, entrypoint_file, resource_file):
        self.clusters = clusters
        self.app = Monolith(icu_file, callgraph_file, entrypoint_file, resource_file)
        self.selected_classes = []
        self.selclass2id = {}

        for cluster_id in clusters.keys():
            classes = clusters[cluster_id]
            self.selected_classes.extend(classes)

        self.selidx2icuidx = {}
        self.icuidx2selidx = {}
        for i,c in enumerate(self.selected_classes):
            self.selclass2id[c] = i
            icuidx = self.app.icuclass2idx[c]
            self.selidx2icuidx[i] = icuidx
            self.icuidx2selidx[icuidx] = i

    def write_to_file(self, output_file):
        clusterjson = []
        edges = []
        nodes = []

        # make nodes
        for idx, name in enumerate(self.selected_classes):
            node = {"entity_type":"class", "label":str(name), "id":str(idx)}
            nodes.append(node)

        icu = self.app.icu

        # make edges
        allrelations = []
        for i,r in enumerate(icu):
            for j,c in enumerate(r):
                if icu[i,j] > 0:
                    r_class = self.icuidx2selidx.get(i)
                    c_class = self.icuidx2selidx.get(j)
                    if r_class is not None and c_class is not None:
                        prop = {"start":str(r_class), "end":str(c_class)}
                        relation = {"label":"icu", "id":str(r_class)+"_"+str(c_class), "properties":prop, "frequency":"1"}
                        allrelations.append(relation)
        edge = {"type":"inter_class_connections", "weight":str(1), "relationship":allrelations}
        edges.append(edge)

        # make clusters
        for clusterid in self.clusters:
            members = self.clusters[clusterid]
            #actual_idxs = [str(filtidx2icuidx[i]) for i in members]
            actual_idxs = [str(self.selclass2id[c]) for c in members]
            #prop = {"nodes":actual_idxs}
            new_cluster_id = str(int(time.time()*10**16))
            c = {"type":"microservices_group", "label":str(clusterid),"id":new_cluster_id, "nodes":actual_idxs}
            clusterjson.append(c)
        #print("Num clusters", len(clusterjson))

        outputjson = {"clusters":clusterjson, "edges":edges, "nodes": nodes}

        with open(output_file, "w") as fj:
            json.dump(outputjson, fj, indent=4)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        #print("Usage: <clusterfile> <icu json> <callgraph file> <entrypoint/service json> <resourcefile> <output filename>")
        sys.exit(1)

    clusterfile = sys.argv[1]
    icufile = sys.argv[2]
    callgraphfile = sys.argv[3]
    servicefile = sys.argv[4]
    resourcefile = sys.argv[5]
    outputfile = sys.argv[6]


    with open(clusterfile, "r") as f:
        clusters = json.load(f)
        #print("Found", len(clusters), "entries in input file")

        formatter = OutputFormatter(clusters, icufile, callgraphfile, servicefile, resourcefile)
        formatter.write_to_file(outputfile)
