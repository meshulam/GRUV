# A labeler function takes a list of blocks with length L, and a name,
# and returns a list of labels also of length L, where each label describes
# block with the same index.

names = []   # TODO: move this state into a Labeler object
def by_name(blocks, name):
    labels = []
    if name not in names:
        names.append(name)

    name_id = names.index(name)
    return [name_id]*len(blocks)

def next_block(blocks, name):
    classes = blocks[1:]
    # 'next' for the last block is zero with whatever shape a block has
    classes.append(blocks[0] * 0)
    return classes

