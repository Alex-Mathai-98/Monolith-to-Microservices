import numpy as np

def read_filecontents(filename, seeds=False):
    """
    Reads the contents of the file line by line
    """
    lines = []
    with open(filename) as f:
        for line in f:
            if seeds:
                lines.append([t for t in line.strip().split()])
            else:
                lines.append(line.strip())
    return lines

def compute_inherit_features(c2imap, inherits_matrix):
    """
    Generates the inheritance feature matrix, keeping into account the classes to ignore
    """
    valid_idxs = [i for i in c2imap.values()]
    # select valid rows from the inherit matrix
    features = inherits_matrix[valid_idxs]
    # using double precision, use float32 if memory is an issue
    features = np.array(features, dtype=np.float64)
    #print("Inheritance features shape:", features.shape)
    return features

def compute_pcc_features(c2imap, cooc_matrix):
    """
    Generates the path class coocurrence feature matrix, keeping into account the classes to ignore
    """
    valid_idxs = [i for i in c2imap.values()]

    features = cooc_matrix[valid_idxs]

    num_zero_features = 0
    scaled_features = []
    for i,row in enumerate(features):
        maxval = np.max(row)
        if maxval == 0:
            num_zero_features += 1
        else:
            row = row / maxval
        scaled_features.append(row)

    # using double precision, use float32 if memory is an issue
    scaled_features = np.array(scaled_features, dtype=np.float64)

    #print("PCC features shape:", scaled_features.shape)
    return scaled_features

def compute_entrypoint_features(c2imap, classes, graph):
    """
    Generates entrypoint feature matrix using the call graph, keeping into account the classes to ignore
    """
    ep2class = graph.entrypoint2class
    class2ep = graph.class2entrypoint

    ep2featureid = {e:i for i,e in enumerate(ep2class.keys())}
    featureid2ep = {i:e for i,e in enumerate(ep2class.keys())}
    num_entrypoints = len(ep2class.keys())

    ep_features = []
    for i,full_classname in enumerate(classes):
        row = [0.0]*num_entrypoints
        num_eps_for_class = 1.0
        entrypoints_for_class = class2ep.get(full_classname)
        if entrypoints_for_class is not None:
            for epname in class2ep[full_classname]:
                idx = ep2featureid[epname]
                row[idx] = 1.0
            num_eps_for_class = np.sum(row)
            row = [r/num_eps_for_class for r in row]
        ep_features.append(row)

    # using double precision, use float32 if memory is an issue
    ep_features = np.array(ep_features, dtype=np.float64)

    #print("EP features shape:", ep_features.shape)
    return ep_features

def analyze_outputs(features, struct_matrix, all_classes):
    """
    Calculate if there are classes that are missing adj matrix connections
    or features. This used to return the intersection, but has been changed
    to return classes that have no features.
    """
    empty_features = np.where(np.sum(features, axis=1) == 0)[0]
    empty_struct = np.where(np.sum(struct_matrix, axis=1) == 0)[0]

    #print("Num classes with no connections:", len(empty_struct))
    
    unsuitable_class_idxs = [i for i in empty_features]
    #print("Num classes with no features:", len(empty_features))
    return unsuitable_class_idxs

def correct_matrices(struct_matrix, all_features, unsuitable_class_idxs):
    (inh_feat, pcc_feat, ep_feat, db_feat) = all_features

    curr_idxs_to_delete = [i for i in unsuitable_class_idxs]
    prev_idxs_to_delete = []
    deleted_idxs = []
    iteration_num = 0
    corrected_features = None

    #print("Before correction struct shapes:", struct_matrix.shape)

    while len(prev_idxs_to_delete) != len(curr_idxs_to_delete):
        iteration_num += 1
        all_corrected_features = []
        #print("Deleting", len(curr_idxs_to_delete), "indices in iteration", iteration_num)

        struct_matrix[:, curr_idxs_to_delete] = 0
        struct_matrix[curr_idxs_to_delete, :] = 0

        if inh_feat is not None:
            inh_feat[:, curr_idxs_to_delete] = 0
            inh_feat[curr_idxs_to_delete, :] = 0
            all_corrected_features.append(inh_feat)
        if pcc_feat is not None:
            pcc_feat[:, curr_idxs_to_delete] = 0
            pcc_feat[curr_idxs_to_delete, :] = 0
            all_corrected_features.append(pcc_feat)
        if ep_feat is not None:
            #ep_feat[:, curr_idxs_to_delete] = 0
            ep_feat[curr_idxs_to_delete, :] = 0
            all_corrected_features.append(ep_feat)
        if db_feat is not None:
            #db_feat[:, curr_idxs_to_delete] = 0
            db_feat[curr_idxs_to_delete, :] = 0
            all_corrected_features.append(db_feat)

        corrected_features = np.concatenate(all_corrected_features, axis=1)
        new_empty_features = np.where(np.sum(corrected_features, axis=1) == 0)[0]

        prev_idxs_to_delete = curr_idxs_to_delete
        curr_idxs_to_delete = [i for i in new_empty_features]

    struct_matrix = np.delete(struct_matrix, curr_idxs_to_delete, axis=0)
    struct_matrix = np.delete(struct_matrix, curr_idxs_to_delete, axis=1)

    if corrected_features is not None:
        corrected_features = np.delete(corrected_features, curr_idxs_to_delete, axis=0)
    else:
        # if no corrections were made, use the orginal features
        features = [f for f in all_features if f is not None]
        corrected_features = np.concatenate(features, axis=1)

    return struct_matrix, corrected_features, curr_idxs_to_delete