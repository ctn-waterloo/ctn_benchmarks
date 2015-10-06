import numpy as np
import nengo

def gather_info(network, inputs, outputs, parents):
    for c in network.connections:
        if c.post_obj not in inputs:
            inputs[c.post_obj] = [c]
        else:
            inputs[c.post_obj].append(c)
        if c.pre_obj not in outputs:
            outputs[c.pre_obj] = [c]
        else:
            outputs[c.pre_obj].append(c)
        parents[c] = network
    for ens in network.ensembles:
        parents[ens] = network
    for n in network.nodes:
        parents[n] = network
    for net in network.networks:
        parents[net] = network
        gather_info(net, inputs, outputs, parents)

def split_passthrough(model, max_dim=16):
    replaced = {}
    inputs = {}
    outputs = {}
    parents = {}
    gather_info(model, inputs, outputs, parents)

    changed = True

    while changed:
      changed = False
      for node in model.all_nodes[:]:
        if node.output is None:
            if node.size_in > max_dim:
                changed = True
                nodes = []
                slices = []
                index = 0
                while index < node.size_in:
                    label = node.label
                    if label is not None:
                        label += ' (%d)' % len(nodes)
                    size = min(node.size_in - index, max_dim)
                    with parents[node]:
                        new_node = nengo.Node(None, size_in=size, label=label)
                        parents[new_node] = parents[node]
                        inputs[new_node] = []
                        outputs[new_node] = []
                    slices.append(slice(index, index + size))
                    nodes.append(new_node)
                    index += size

                for c in inputs.get(node, [])[:]:
                    base_transform = c.transform
                    if len(base_transform.shape) == 0:
                        base_transform = np.eye(c.size_mid) * base_transform
                    transform = np.zeros((node.size_in, c.size_in))
                    transform[c.post_slice] = base_transform

                    for i, n in enumerate(nodes):
                        t = transform[slices[i]]
                        if np.count_nonzero(t) > 0:
                            with parents[c]:
                                new_c = nengo.Connection(c.pre, n,
                                                 transform=t,
                                                 function=c.function,
                                                 synapse=c.synapse)
                                inputs[n].append(new_c)
                                outputs[c.pre_obj].append(new_c)
                                parents[new_c] = parents[c]
                    outputs[c.pre_obj].remove(c)
                    inputs[node].remove(c)
                    parents[c].connections.remove(c)

                for c in outputs.get(node, [])[:]:
                    base_transform = c.transform
                    if len(base_transform.shape) == 0:
                        base_transform = np.eye(c.size_mid) * base_transform
                    transform = np.zeros((c.size_out, node.size_out))
                    transform[:, c.pre_slice] = base_transform

                    for i, n in enumerate(nodes):
                        t = transform[:, slices[i]]
                        if np.count_nonzero(t) > 0:
                            with parents[c]:
                                new_c = nengo.Connection(n, c.post,
                                                 transform=t,
                                                 synapse=c.synapse)
                                outputs[n].append(new_c)
                                inputs[c.post_obj].append(new_c)
                                parents[new_c] = parents[c]
                    inputs[c.post_obj].remove(c)
                    outputs[node].remove(c)
                    parents[c].connections.remove(c)

                parents[node].nodes.remove(node)
                replaced[node] = nodes

    for p in model.probes[:]:
        if p.target in replaced:
            model.probes.remove(p)
            probes = []
            for n in replaced[p.target]:
                with model:
                    probe = nengo.Probe(n, synapse=p.synapse)
                probes.append(probe)
            replaced[p] = probes

    return replaced
