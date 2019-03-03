'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,

            batch_size=1,
            epochs=1,

            lr=6.25e-5,
            lr_schedule='warmup_linear',
            lr_warmup=0.002,
            b1=0.9,
            b2=0.999,
            e=1e-8,
            l2=0.01,
            vector_l2=False,
            max_grad_norm=1,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        # train
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_warmup = lr_warmup
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.l2 = l2
        self.vector_l2 = vector_l2
        self.max_grad_norm = max_grad_norm

    def __getitem__(self, x):
        return self.__getattribute__(x)
