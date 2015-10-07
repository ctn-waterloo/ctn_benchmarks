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


def split_input_nodes(model, max_dim=16):
    replaced = {}
    inputs = {}
    outputs = {}
    parents = {}
    gather_info(model, inputs, outputs, parents)

    for node in model.all_nodes[:]:
        if node.size_in == 0 and node.size_out > max_dim:
            index = 0
            nodes = []
            slices = []
            function = node.output

            while index < node.size_out:
                size = min(node.size_out - index, max_dim)
                label = node.label
                if label is not None:
                    label += '[%d:%d]' % (index, index + size)

                with parents[node]:
                    n = nengo.Node(node.output, size_in=0,
                                   size_out=node.size_out,
                                   label=label)
                    nodes.append(n)
                    slices.append(slice(index, index + size))
                index += size

            for c in outputs[node]:
                assert c.pre_slice == slice(None, None)
                assert c.transform == 1.0

                with parents[c]:
                    for i, n in enumerate(nodes):
                        nengo.Connection(n[slices[i]],
                                         c.post[slices[i]],
                                         synapse=c.synapse,
                                         )
                parents[c].connections.remove(c)
            parents[node].nodes.remove(node)








def pass_ensembles(model, max_dim=16):
    replaced = {}
    inputs = {}
    outputs = {}
    parents = {}
    gather_info(model, inputs, outputs, parents)

    for ens in model.all_ensembles[:]:
        total_out = {}
        conns = {}
        for c in outputs[ens]:
            if isinstance(c.post, nengo.ensemble.Neurons):
                continue
            if c.function is None:
                key = None, None
            else:
                key = c.function, repr(c.pre_slice)
            if key not in total_out:
                total_out[key] = c.size_out
                conns[key] = [c]
            else:
                total_out[key] += c.size_out
                conns[key].append(c)

        for key, total in total_out.items():
            if total > max_dim:
                f, slice = key
                cs = conns[key]
                with parents[ens]:
                    node = nengo.Node(None, size_in=cs[0].size_mid)
                    pre = ens if slice is None else ens[cs[0].pre_slice]
                    nengo.Connection(pre, node, synapse=None, function=f)
                for c in cs:
                    with parents[c]:
                        nengo.Connection(node, c.post, synapse=c.synapse,
                                         transform=c.transform)
                    parents[c].connections.remove(c)


def remove_outputless_passthrough(model):
    replaced = {}
    inputs = {}
    outputs = {}
    parents = {}
    gather_info(model, inputs, outputs, parents)

    for node in model.all_nodes[:]:
        if node.output is None and node not in outputs:
            for p in model.probes:
                if p.target is node:
                    break
            else:
                for c in inputs[node]:
                    parents[c].connections.remove(c)
                parents[node].nodes.remove(node)



