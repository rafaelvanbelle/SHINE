import random
import torch
import numpy as np
import operator

from torch_geometric.sampler.utils import (
    remap_keys
)

def custom_sample_fn(colptr, row_data, index, num_neighbors, replace=False, directed=True):

    if not isinstance(index, torch.LongTensor):
        index = torch.LongTensor(index)

    
    samples = []
    to_local_node = {}

    for i in range(len(index)):
        v = index[i]
        samples.append(v)
        to_local_node[v.item()] = i


    rows = []
    cols = []
    edges = []

    begin = 0
    end = len(samples.copy())

    for ell in range(len(num_neighbors)):
        num_samples = num_neighbors[ell]
        for i in range(begin, end):
            w = samples[i]
            col_start = colptr[w]
            col_end = colptr[w+1]
            col_count = col_end - col_start
            if (col_count == 0):
                continue

            # only transactions are target nodes. Hence, their first order neighbors are always cardholder
            # and merchant! We need to skip these!
            #cardholder = row_data[col_start]
            #merchant = row_data[col_end]

        
            
            if ((num_samples < 0) or (not replace and (num_samples >= col_count))): 
                for offset in range(col_start,col_end):
                    v = row_data[offset];
                    if v.item() not in to_local_node.keys():
                        to_local_node[v.item()] = len(samples)
                        samples.append(v)
                    if (directed):
                            cols.append(i);
                            rows.append(to_local_node[v.item()]);
                            edges.append(offset);
                            
            elif (replace):
                for j in range(num_samples):
                    pass
                    print('warning replace is not implemented!! See code.')
                    #const int64_t offset = col_start + rand() % col_count;
                    #const int64_t &v = row_data[offset];
                    #const auto res = to_local_node.insert({v, samples.size()});
                    #if (res.second)
                    #    samples.push_back(v);
                    #if (directed) {
                    #    cols.push_back(i);
                    #    rows.push_back(res.first->second);
                    #    edges.push_back(offset);
                    
            else:
                rnd_indices = [] 
                j = col_count - num_samples
                for j in range((col_count - num_samples), col_count):
                    rnd = random.randint(0,j);
                    if (rnd in rnd_indices):
                        rnd = j
                        rnd_indices.append(j);
            
                    offset = col_start + rnd
                    v = row_data[offset]

                    if v.item() not in to_local_node.keys():
                        to_local_node[v.item()] = len(samples)
                        samples.append(v)
                    if (directed): 
                        cols.append(i)
                        rows.append(to_local_node[v.item()]) #res.first->second)
                        edges.append(offset.item());

        begin = end
        end = len(samples)

        
    if (not directed):

        #unordered_map<int64_t, int64_t>::iterator iter;
        
        for i in range(len(samples)):
            w = samples[i]
            col_start = colptr[w]
            col_end = colptr[w + 1]

            for offset in range(col_start, col_end):

                v = row_data[offset]
                if v.item() in to_local_node.keys():
        
                    rows.append(to_local_node[v.item()])
                    cols.append(i)
                    edges.append(offset)


    samples = torch.stack(samples) 
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    edges = torch.tensor(edges)

    return samples, rows, cols, edges




def custom_weighted_sample_fn(colptr, row_data, edge_weight, index, num_neighbors, replace=False, directed=True):

    if not isinstance(index, torch.LongTensor):
        index = torch.LongTensor(index)

    
    samples = []
    to_local_node = {}

    for i in range(len(index)):
        v = index[i]
        samples.append(v)
        to_local_node[v.item()] = i


    rows = []
    cols = []
    edges = []

    
    begin = 0
    end = len(samples.copy())

    for ell in range(len(num_neighbors)):
        num_samples = num_neighbors[ell]
        for i in range(begin, end):
            w = samples[i]
            col_start = colptr[w]
            col_end = colptr[w+1]
            col_count = col_end - col_start
            if (col_count == 0):
                continue

            # only transactions are target nodes. Hence, their first order neighbors are always cardholder
            # and merchant! We need to skip these!
            #cardholder = row_data[col_start]
            #merchant = row_data[col_end]

        
            
            if ((num_samples < 0) or (not replace and (num_samples >= col_count))): 
                for offset in range(col_start,col_end):
                    v = row_data[offset];
                    if v.item() not in to_local_node.keys():
                        to_local_node[v.item()] = len(samples)
                        samples.append(v)
                    if (directed):
                            cols.append(i);
                            rows.append(to_local_node[v.item()]);
                            edges.append(offset);
                            
            elif (replace):
                for j in range(num_samples):
                    pass
                    print('warning replace is not implemented!! See code.')
                    #const int64_t offset = col_start + rand() % col_count;
                    #const int64_t &v = row_data[offset];
                    #const auto res = to_local_node.insert({v, samples.size()});
                    #if (res.second)
                    #    samples.push_back(v);
                    #if (directed) {
                    #    cols.push_back(i);
                    #    rows.push_back(res.first->second);
                    #    edges.push_back(offset);

            # Sampling without replacement.        
            else:
                columns = range(col_start, col_end)
                weights = edge_weight[col_start:col_end]
            
                weights= np.asarray(weights).astype('float64')
                weights = (weights)/np.sum(weights)

                neighbors = row_data[col_start:col_end]
                sampled_columns = np.random.choice(columns, size=num_samples, replace=False, p=weights)
                
                for offset in sampled_columns:

                    v = row_data[offset]

                    if v.item() not in to_local_node.keys():
                        to_local_node[v.item()] = len(samples)
                        samples.append(v)
                    if (directed): 
                        cols.append(i)
                        rows.append(to_local_node[v.item()]) #res.first->second)
                        edges.append(offset.item());


                    

        begin = end
        end = len(samples)

        
    if (not directed):

        #unordered_map<int64_t, int64_t>::iterator iter;
        
        for i in range(len(samples)):
            w = samples[i]
            col_start = colptr[w]
            col_end = colptr[w + 1]

            for offset in range(col_start, col_end):

                v = row_data[offset]
                if v.item() in to_local_node.keys():
        
                    rows.append(to_local_node[v.item()])
                    cols.append(i)
                    edges.append(offset)


    samples = torch.stack(samples) 
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    edges = torch.tensor(edges)

    return samples, rows, cols, edges



def custom_skip_sample_fn(colptr, row_data, edge_weight, index, num_neighbors, replace=False, directed=True, weight_func = 'mul', exp=False):

    """Custom skip sample fn is a neighbor sampling function with weighted probabilities. The neighbors are sampled 
    two layers at a time. 
    
    For the source node, both the first order neighbors and second order neighbors are sampled at once. 
    The first order neighbors are selected all (no sampling). Then, the second order neighbors are sampled with weighted probabilities. 
    The weights are determined by the combination of the first order neighbor edge weight and the second order edge weight. 

    source --*edge weight1*--> first order neighbor --*edge weight2*--> second order neighbor

    The sampling probability associated with second order neighbor is: operator(edge_weight_2, edge_weight_1) 
    The operator can be passed as an argument to this function.

    Returns:
        _type_: _description_
    """

    if not isinstance(index, torch.LongTensor):
        index = torch.LongTensor(index)

    num_hops = len(num_neighbors)
    
    samples = []
    to_local_node = {}

    for i in range(len(index)):
        v = index[i]
        samples.append(v)
        to_local_node[v.item()] = i


    rows = []
    cols = []
    edges = []

    begin = 0
    end = len(samples.copy())

    #print("we have ", end, " source nodes")

    #####!!!!!#### For the weighted version we do provide the number of neighbors for two layers [2,32], 
    # but in this skip version we only do one iteration (we sample both layers in one go)
    # hence the /2 
    for ell in range(int(num_hops/2)):
    #for ell in range(len(num_neighbors)):
        # ell1 is outer hop
        # ell2 is inner hop
        ell1 = 2*ell
        ell2 = 2*ell + 1
        #print(ell1)
        #print(ell2)
        
        num_samples = num_neighbors[ell1]
        for i in range(begin, end):
            
            #print("we start an outer loop")
            # get cardholder merchant and add them to samples and local_node, add them to cols, rows, edges
            # log for both cardholder and merchant the source node
            # Now we immediatly sample for cardholder and merchant. In this way, we ensure that we now the source of the second stage

            #print("i", str(i))

            inner_begin = len(samples)
            
            w = samples[i]
            col_start = colptr[w]
            col_end = colptr[w+1]
            col_count = col_end - col_start

            #print("we add all first order neighbors")

            if (col_count == 0):
                continue

            for offset in range(col_start,col_end):
                v = row_data[offset];
                if v.item() not in to_local_node.keys():
                    #print("we add node ", v, " that is a neighbor of ", w)
                    #print("associated offset ", offset)
                    to_local_node[v.item()] = len(samples)
                    samples.append(v)
                if (directed):
                        cols.append(i);
                        rows.append(to_local_node[v.item()]);
                        edges.append(offset);

                #print(edge_weight[offset])
            inner_end = len(samples)
            #print("inner begin", str(inner_begin))
            #print("inner end", str(inner_end))

            #print("we start an inner loop")

            for j in range(inner_begin,inner_end):
                #print("j", str(j))
                #print('col_start', col_start)
                # inner_begin and inner_end determine how many neighbors the source node w has. 
                # The inner offset determines where we can find the information of a particular neighbor of w in the orginal data (e.g. edge weight data)
                inner_offset = col_start + (j - inner_begin)
                #print("inner_offset", str(inner_offset))

                # This is the weight associated with the edge between source node w and neighbor x
                incoming_edge_weight = edge_weight[inner_offset]
                #print("incoming_edge_weight", str(incoming_edge_weight))

                x = samples[j]
                inner_col_start = colptr[x]
                inner_col_end = colptr[x+1]
                inner_col_count = inner_col_end - inner_col_start
                inner_num_samples = ell2
                #print('inner_col_count', str(inner_col_count))
                #print('num_samples', str(num_samples))

                # Retrieve all neighbor columns
                columns = range(inner_col_start, inner_col_end)
                weights = edge_weight[inner_col_start:inner_col_end]
                #print("original weight", str(weights[0]))
                # retrieve the operator to apply to the edge weights of the neighbors and the incoming edge weight.
                op = getattr(operator, weight_func)
                # Get all the weights from the neighbors and apply an operation (multiply, deduct,.. )with the incoming edge weight
                weights = op(incoming_edge_weight, edge_weight[inner_col_start:inner_col_end])
                # make sure we have no negative weights. All negatives are put to zero.
                #torch.nn.functional.relu(weights, inplace=True)


                weights = torch.abs(weights)

                if weight_func == 'sub':
                    # neighbors with edge weights close to the incoming edge weight are favoured.
                    weights = torch.max(weights) - weights

                #print("new weight", str(weights[0]))
                # Only sample the number of neighbors that have a non zero weight if there are less than num_samples
                inner_col_count = torch.count_nonzero(weights)
                #print('inner col count', str(inner_col_count))

                if exp:
                    weights = np.exp(weights)

                if (inner_col_count == 0):
                    continue

                if ((num_samples < 0) or (not replace and (inner_num_samples >= inner_col_count))): 
                    for offset in range(inner_col_start,inner_col_end):
                        v = row_data[offset];
                        if v.item() not in to_local_node.keys():
                            #print("we add node ", v, " that is a neighbor of ", x)
                            to_local_node[v.item()] = len(samples)
                            samples.append(v)
                        if (directed):
                            cols.append(j);
                            rows.append(to_local_node[v.item()]);
                            edges.append(offset);
                            #print("offset", offset)
                                
                elif (replace):
                    for k in range(inner_num_samples):
                        pass
                        print('warning replace is not implemented!! See code.')
                        #const int64_t offset = col_start + rand() % col_count;
                        #const int64_t &v = row_data[offset];
                        #const auto res = to_local_node.insert({v, samples.size()});
                        #if (res.second)
                        #    samples.push_back(v);
                        #if (directed) {
                        #    cols.push_back(i);
                        #    rows.push_back(res.first->second);
                        #    edges.push_back(offset);
                        
                else:
                    
                    
                    
                    #print("incoming weight", str(incoming_edge_weight))
                    #print("modified weight", str(weights[0]))

                    # Normalize weights so they sum to 1. (turn into probabilities)
                    weights= np.asarray(weights).astype('float64')
                    weights = (weights)/np.sum(weights)
                

                        
                    #neighbors = row_data[inner_col_start:inner_col_end]
                    sampled_columns = np.random.choice(columns, size=inner_num_samples, replace=False, p=weights)
                
                    for offset in sampled_columns:

                        v = row_data[offset]

                        if v.item() not in to_local_node.keys():
                            to_local_node[v.item()] = len(samples)
                            samples.append(v)
                            #print("we add node ", v, " that is a neighbor of ", x)
                        if (directed): 
                            cols.append(j)
                            rows.append(to_local_node[v.item()]) #res.first->second)
                            edges.append(offset.item());
                            #print("offset", offset)

                
        begin = end
        end = len(samples)
        #print('new_end', str(end))
        
    if (not directed):
        #print("not directed")
        #unordered_map<int64_t, int64_t>::iterator iter;
        
        for i in range(len(samples)):
            w = samples[i]
            col_start = colptr[w]
            col_end = colptr[w + 1]

            for offset in range(col_start, col_end):

                v = row_data[offset]
                if v.item() in to_local_node.keys():
        
                    rows.append(to_local_node[v.item()])
                    cols.append(i)
                    edges.append(offset)



    samples = torch.stack(samples) 
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    edges = torch.tensor(edges)

    return samples, rows, cols, edges


## HeteroSample
    

    
def custom_hetero_sample_fn(node_types, 
              edge_types, 
              colptr_dict, 
              row_dict, 
              input_node_dict, 
              num_neighbors_dict, 
              num_hops, 
              replace=False, 
              directed=True,
              labeled=False):
    
    # Create a mapping to convert single string relations to edge type triplets:
    to_edge_type = {}
    for k in edge_types:
        to_edge_type[k[0] + "__" + k[1] + "__" + k[2]] = (k[0], k[1], k[2])
    
    #// Initialize some data structures for the sampling process:
    samples_dict = {}
    to_local_node_dict = {}
    for node_type in node_types:
        samples_dict[node_type] = []
        to_local_node_dict[node_type] = {}
        
    rows_dict = {}
    cols_dict = {}
    edges_dict = {}
    for key in colptr_dict.keys():
        rel_type = key
        rows_dict[rel_type] = []
        cols_dict[rel_type] = []
        edges_dict[rel_type] = []
        
    #// Add the input nodes to the output nodes:
    for key, value in input_node_dict.items(): 
        node_type = key
        input_node_data = value
        #input_node_data = input_node.data_ptr<int64_t>();


        for i in range(len(input_node_data)):
            v = input_node_data[i];
            samples_dict[node_type].append(v.item())
            to_local_node_dict[node_type][v.item()] = i 

    slice_dict = {}
    for key,value in samples_dict.items():
        slice_dict[key] = (0, len(value))



        
    for ell in range(num_hops):
        #print(ell)
        for key, value in num_neighbors_dict.items(): 
            #print('num_neighbor_key', str(key))
            #print('num_neighbor_value', str(value))
            rel_type = key
            edge_type = to_edge_type[rel_type]
            src_node_type = edge_type[0]
            dst_node_type = edge_type[2]
            #print('src_node_type', str(src_node_type))
            #print('dst_node_type', str(dst_node_type))
            num_samples = value[ell]
            dst_samples = samples_dict[dst_node_type]
            src_samples = samples_dict[src_node_type]
            #print('dst_samples', str(dst_samples))
            #print('src_samples', str(src_samples))
            to_local_src_node = to_local_node_dict[src_node_type]

            colptr = colptr_dict[rel_type]
            row_data = row_dict[rel_type]

            rows = rows_dict[rel_type]
            cols = cols_dict[rel_type]
            edges = edges_dict[rel_type]

            # Each node type might have a differen number of total nodes
            begin = slice_dict[dst_node_type][0]
            end = slice_dict[dst_node_type][1]

            #print(str(dst_node_type))
            #print('begin', str(begin))
            #print('end', str(end))
            for i in range(begin,end):
                w = dst_samples[i]
                col_start = colptr[w]
                col_end = colptr[w + 1]
                col_count = col_end - col_start
                #print('col_count', str(col_count))
                if (col_count == 0):
                    #print('no neighbors to sample from')
                    continue

            # voeg alle neighbors (src_samples) toe

            # voor de 

            # for inner_key, inner_value in num_neighbors_dict.items():
                #inner_rel_type = inner_key
                #inner_edge_type = to_edge_type[inner_rel_type]
                #inner_src_node_type = inner_edge_type[0]
                #inner_dst_node_type = inner_edge_type[2]
                #if inner_dst_node_type != src_node_type:
                    continue
                #if inner_dst_node_type == src_node_type:





                if ((num_samples < 0) or ((not replace) and (num_samples >= col_count))):
                    for offset in range(col_start, col_end): 
                        v = row_data[offset]

                        if v.item() not in to_local_src_node.keys():
                            to_local_src_node[v.item()] = len(src_samples)
                            src_samples.append(v.item())
                        if (directed):
                            if labeled & (src_node_type == 'transaction') & (to_local_src_node[v.item()] < len(input_node_data)):
                                pass
                            else:
                                cols.append(i)
                                rows.append(to_local_src_node[v.item()])
                                edges.append(offset)
                            #print("offset", str(offset))

                elif (replace):
                    for j in range(num_samples): 
                        print('warning replace is not implemented!! See code.')
                        pass
                        #offset = col_start + rand() % col_count;
                        #v = row_data[offset];
                        #if v.item() not in to_local_src_node.keys():
                        #    to_local_src_node[v.item()] = len(src_samples)
                        #    samples.append(v)

                        #if (directed): 
                        #  cols.append(i)
                        #  rows.append(to_local_node[v.item()])
                        #  edges.append(offset)


                else:
                    rnd_indices = []
                    for j in range((col_count - num_samples), col_count):

                        rnd = random.randint(0,j);
                        if (rnd in rnd_indices):
                            rnd = j
                            rnd_indices.append(j)
                        else:
                            rnd_indices.append(rnd)


                        offset = int(col_start) + rnd
                        v = row_data[offset]
                        if v.item() not in to_local_src_node.keys():
                            to_local_src_node[v.item()] = len(src_samples)
                            src_samples.append(v.item())
                        if (directed):
                            if labeled & (src_node_type == 'transaction') & (to_local_src_node[v.item()] < len(input_node_data)):
                                pass
                            else:
                                cols.append(i)
                                rows.append(to_local_src_node[v.item()])
                                edges.append(offset)
                            #print("offset", str(offset))

        for key, value in samples_dict.items():
            slice_dict[key] = (slice_dict[key][1], len(value))

    #print(samples_dict)
    for dic in [samples_dict, rows_dict, cols_dict, edges_dict]:
        for k,v in dic.items():
            dic[k] = torch.tensor(v).to(torch.long)
    
    print("remapping")

    rows_dict=remap_keys(rows_dict, to_edge_type)
    cols_dict=remap_keys(cols_dict, to_edge_type)
    edges_dict=remap_keys(edges_dict, to_edge_type)

    print(rows_dict.keys())
    return samples_dict, rows_dict, cols_dict, edges_dict


def custom_skip_hetero_sample_fn(node_types, 
              edge_types, 
              colptr_dict, 
              row_dict,
              edge_weight_dict, 
              input_node_dict, 
              num_neighbors_dict, 
              num_hops, 
              replace=False, 
              directed=True,
              labeled=False, 
              weight_func = 'mul',
              exp = False):
    
    # Create a mapping to convert single string relations to edge type triplets:
    to_edge_type = {}
    for k in edge_types:
        to_edge_type[k[0] + "__" + k[1] + "__" + k[2]] = (k[0], k[1], k[2])
    
    #// Initialize some data structures for the sampling process:
    samples_dict = {}
    to_local_node_dict = {}
    for node_type in node_types:
        samples_dict[node_type] = []
        to_local_node_dict[node_type] = {}
        
    rows_dict = {}
    cols_dict = {}
    edges_dict = {}
    for key in colptr_dict.keys():
        rel_type = key
        rows_dict[rel_type] = []
        cols_dict[rel_type] = []
        edges_dict[rel_type] = []
        
    #// Add the input nodes to the output nodes:
    for key, value in input_node_dict.items(): 
        node_type = key
        input_node_data = value
        #input_node_data = input_node.data_ptr<int64_t>();


        for i in range(len(input_node_data)):
            v = input_node_data[i];
            samples_dict[node_type].append(v.item())
            to_local_node_dict[node_type][v.item()] = i 

    slice_dict = {}
    for key,value in samples_dict.items():
        slice_dict[key] = (0, len(value))


    inner_slice_dict = slice_dict.copy()

    #####!!!!!#### For the hetero version we do provide the number of neighbors for two layers [2,32], 
    # but in this skip version we only do one iteration (we sample both layers in one go)
    # hence the /2 
    for ell in range(int(num_hops/2)):
        # ell1 is outer hop
        # ell2 is inner hop
        ell1 = 2*ell
        ell2 = 2*ell + 1
        #print(ell1)
        #print(ell2)

        for key, value in num_neighbors_dict.items(): 
            #print('num_neighbor_key', str(key))
            #print('num_neighbor_value', str(value))
            rel_type = key
            edge_type = to_edge_type[rel_type]
            src_node_type = edge_type[0]
            dst_node_type = edge_type[2]
            #print('src_node_type', str(src_node_type))
            #print('dst_node_type', str(dst_node_type))
            num_samples = value[ell1]
            dst_samples = samples_dict[dst_node_type]
            src_samples = samples_dict[src_node_type]
            
            to_local_src_node = to_local_node_dict[src_node_type]

            colptr = colptr_dict[rel_type]
            row_data = row_dict[rel_type]
            edge_weight = edge_weight_dict[rel_type]

            rows = rows_dict[rel_type]
            cols = cols_dict[rel_type]
            edges = edges_dict[rel_type]

            # Each node type might have a differen number of total nodes
            begin = slice_dict[dst_node_type][0]
            end = slice_dict[dst_node_type][1]

            #print(str(dst_node_type))
            #print('begin', str(begin))
            #print('end', str(end))

            # hier voeg je alle first order neighbors toe aan de output
            for i in range(begin,end):

                #print("current i ", i)
                w = dst_samples[i]
                #print("source node is: ", w)
                col_start = colptr[w]
                col_end = colptr[w + 1]
                col_count = col_end - col_start
                #print("source node has: ", col_count, " neighbors")
                #print('col_count', str(col_count))
                if (col_count == 0):
                    #print('no neighbors to sample from')
                    continue

            # voeg alle neighbors (src_samples) toe

            ####### Voeg alle neighbors 
                for offset in range(col_start, col_end): 
                    v = row_data[offset]

                    if v.item() not in to_local_src_node.keys():
                        to_local_src_node[v.item()] = len(src_samples)
                        src_samples.append(v.item())
                    if (directed):
                        cols.append(i)
                        rows.append(to_local_src_node[v.item()])
                        edges.append(offset)

                        #print("offset", str(offset))

                # the inner slice dict is updated with the first order neighbors. This allows the second loop (see below)
                # to use these first order neighbors to search for second order neighbors.
                inner_slice_dict[src_node_type] = (inner_slice_dict[src_node_type][1], len(src_samples))
                #print("inner_slice_dict", inner_slice_dict)


                for inner_key, inner_value in num_neighbors_dict.items():
                    #print('inner_num_neighbor_key', str(inner_key))
                    #print('inner_num_neighbor_value', str(inner_value))
                    inner_rel_type = inner_key
                    inner_edge_type = to_edge_type[inner_rel_type]
                    inner_src_node_type = inner_edge_type[0]
                    inner_dst_node_type = inner_edge_type[2]
                    
                    
                    inner_num_samples = inner_value[ell2]
                    inner_dst_samples = samples_dict[inner_dst_node_type]
                    inner_src_samples = samples_dict[inner_src_node_type]
                    
                    #print("inner_num_samples", inner_num_samples)
                    # for the inner loop we re-iterate over all possible edge types. BUT we only continue with those
                    # edge types for which the inner_dst_node_type is equal to the src_node type of the outer loop.
                    # in other words, the nodes for which we search neighbors in this loop should be of the same type as 
                    # the neighbors we retrieved in the outer loop. 

                    if inner_dst_node_type != src_node_type:
                        #print('inner_dst_node != src_node')
                        continue

                    if inner_dst_node_type == src_node_type:
                        #print('inner_dst_node == src_node')
                        #print('inner_src_node_type', str(inner_src_node_type))
                        #print('inner dst node type moet hier een cardholder of merchant zijn!')
                        #print('inner_dst_node_type', str(inner_dst_node_type))
                        inner_to_local_src_node = to_local_node_dict[inner_src_node_type]

                        inner_colptr = colptr_dict[inner_rel_type]
                        inner_row_data = row_dict[inner_rel_type]
                        inner_edge_weight = edge_weight_dict[inner_rel_type]

                        inner_rows = rows_dict[inner_rel_type]
                        inner_cols = cols_dict[inner_rel_type]
                        inner_edges = edges_dict[inner_rel_type]

                        # Each node type might have a differen number of total nodes
                        inner_begin = inner_slice_dict[inner_dst_node_type][0]
                        inner_end = inner_slice_dict[inner_dst_node_type][1]

                        #print(str(inner_dst_node_type))
                        #print('inner begin', str(inner_begin))
                        #print('inner end', str(inner_end))
                        for j in range(inner_begin,inner_end):
                            
                            # inner_begin and inner_end determine how many neighbors the source node  has. 
                            # The inner offset determines where we can find the information of a particular neighbor of w in the orginal data (e.g. edge weight data)
                            inner_offset = col_start + (j - inner_begin)
                            #print("inner_offset", inner_offset)

                            # This is the weight associated with the edge between source node w and neighbor x
                            incoming_edge_weight = edge_weight[inner_offset]
                            #print("incoming_edge_weight", str(incoming_edge_weight))

                            v = inner_dst_samples[j]
                            #print("node type second order neighbor: ", inner_src_node_type)
                            #print("node type first order neighbor: ", inner_dst_node_type)

                            inner_col_start = inner_colptr[v]
                            inner_col_end = inner_colptr[v + 1]
                            inner_col_count = inner_col_end - inner_col_start
                            #print('inner_col_count', str(inner_col_count))
                            

                            # Retrieve all neighbor columns
                            columns = range(inner_col_start, inner_col_end)
                            weights = inner_edge_weight[inner_col_start:inner_col_end]
                            # retrieve the operator to apply to the edge weights of the neighbors and the incoming edge weight.
                            op = getattr(operator, weight_func)
                            # Get all the weights from the neighbors and apply an operation (multiply, deduct,.. )with the incoming edge weight
                            weights = op(incoming_edge_weight, inner_edge_weight[inner_col_start:inner_col_end])
                            
                            # make sure we have no negative weights. All negatives are put to zero.
                            #torch.nn.functional.relu(weights, inplace=True)

                            weights = torch.abs(weights)
                            #print(weights)
                            if (weight_func == 'sub') & len(weights) > 1:
                                #print('we have more than one')
                                # neighbors with edge weights close to the incoming edge weight are favoured.
                                weights = torch.max(weights) - weights

                            #print(weights)
                            #print("new weight", str(weights[0]))
                            # Only sample the number of neighbors that have a non zero weight if there are less than num_samples
                            inner_col_count = torch.count_nonzero(weights)
                            #print('inner col count after weights', str(inner_col_count))
                            if exp:
                                weights = np.exp(weights)
                            
                            if (inner_col_count == 0):
                                #print('no neighbors to sample from')
                                continue
                           # print('inner_col_count', str(inner_col_count))


                            if ((inner_num_samples < 0) or ((not replace) and (inner_num_samples >= inner_col_count))):
                                #print("inner_col_count is smaller than inner_num_samples")
                                
                                for offset in range(inner_col_start, inner_col_end): 
                                    
                                    x = inner_row_data[offset]
                                    #print('first order neighbor v ', v,  ' has a neighbor x ', x)
                                    if x.item() not in inner_to_local_src_node.keys():
                                        inner_to_local_src_node[x.item()] = len(inner_src_samples)
                                        inner_src_samples.append(x.item())
                                    if (directed):
                                        inner_cols.append(j)
                                        inner_rows.append(inner_to_local_src_node[x.item()])
                                        inner_edges.append(offset)
                                        #print("offset", str(offset))

                            elif (replace):
                                for j in range(num_samples): 
                                    print('warning replace is not implemented!! See code.')
                                    pass
                                    #offset = col_start + rand() % col_count;
                                    #v = row_data[offset];
                                    #if v.item() not in to_local_src_node.keys():
                                    #    to_local_src_node[v.item()] = len(src_samples)
                                    #    samples.append(v)

                                    #if (directed): 
                                    #  cols.append(i)
                                    #  rows.append(to_local_node[v.item()])
                                    #  edges.append(offset)


                            else:

                                # Normalize weights so they sum to 1. (turn into probabilities)
                                weights= np.asarray(weights).astype('float64')
                                weights = (weights)/np.sum(weights)
                                
                                #neighbors = row_data[inner_col_start:inner_col_end]
                                sampled_columns = np.random.choice(columns, size=inner_num_samples, replace=False, p=weights)
                                #print(sampled_columns)

                                for offset in sampled_columns:

                                    x = inner_row_data[offset]

                                    if x.item() not in inner_to_local_src_node.keys():
                                        inner_to_local_src_node[x.item()] = len(inner_src_samples)
                                        inner_src_samples.append(x.item())
                                    if (directed):
                                        inner_cols.append(j)
                                        inner_rows.append(inner_to_local_src_node[x.item()])
                                        inner_edges.append(offset)
                                        #print("offset", str(offset))
                                

                    # for every edge type in the second layer, we make sure that the dst node (first order neighbor) gets updated in the slice dict.
                    # If our model is deeper, we don't want to sample neighbors for the first layer anymore, we already did this above!
                    inner_slice_dict[inner_dst_node_type] = (inner_slice_dict[inner_dst_node_type][1], len(samples_dict[inner_dst_node_type]))
                
                #for key, value in samples_dict.items():
                #    print("key", key)
                #    print("value len", len(value))
                #    inner_slice_dict[key] = (inner_slice_dict[key][1], len(value))
                #print("intermediate update slide dict", inner_slice_dict)

            ### ADDED ###
            
            #for key, value in samples_dict.items():
            #    print("key", key)
            #    print("value len", len(value))
            #    inner_slice_dict[key] = (inner_slice_dict[key][1], len(value))

            #print("intermediate update slide dict", inner_slice_dict)
                                    #####

        # after each two layers (two layers are sampled at once), the values for the slice dict are updated. 
        for key, value in samples_dict.items():

            slice_dict[key] = (inner_slice_dict[key][1], len(value))

        #print("final update slide dict", slice_dict)
    #print(samples_dict)
    for dic in [samples_dict, rows_dict, cols_dict, edges_dict]:
        for k,v in dic.items():
            dic[k] = torch.tensor(v).to(torch.long)
    

    return samples_dict, rows_dict, cols_dict, edges_dict
