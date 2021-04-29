import torch

def contact_map_convert():
    print(1)


def build_collate_fn(dataset_type):
    def collate_fn(batch):
        if len(batch[0]) == 2:
            data, batch_converters, = zip(*batch)
            batch_converter = batch_converters[0]
            labels, strs, tokens = batch_converter(data)
            contact_maps = None
            paths = None
        elif len(batch[0]) == 3:
            data, contact_maps, batch_converters, = zip(*batch)
            batch_converter = batch_converters[0]
            labels, strs, tokens = batch_converter(data)
            contact_maps = [torch.Tensor(contact_map) for contact_map in contact_maps]
            paths = None
        else:
            raise Exception("Unexpected Num of Components in a Batch")

        return tokens, contact_maps, labels, strs, paths

    return collate_fn



