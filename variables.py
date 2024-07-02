batch_size = 64 # int(args.batch_size) # 128 makes no sense, too fat, and the slope is not steep at all
block_size = 256 # T of the model. how big of a char sequence is taken each iter
max_iters = 12000
learning_rate = 3e-5 # can become smaller with scheduler influencing it for a sharper result
eval_iters = 50 # iterations done for loss calculation
n_embd = 512 # embedding table. 512 is franqly pretty big
n_head = 6 # is enough. For BIG ones 12+, rarest cases for real fat ones is 96
n_layer = 8
dropout = 0.22
save_iters = 1000
vocab_to_use = 'project_data/vocab.txt'
files_to_use = [
    'project_data/wizard_of_jules_short.txt',
    'project_data/wizard_od_jules_short1.txt',
    'project_data/wizard_od_jules_short2.txt',
    'project_data/wizard_od_jules_short3.txt'
]