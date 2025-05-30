# import torch
# import torch.nn as nn
# from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

# __all__ = ['BertTextEncoder']

# TRANSFORMERS_MAP = {
#     'bert': (BertModel, BertTokenizer),
#     'roberta': (RobertaModel, RobertaTokenizer),
# }

# class BertTextEncoder(nn.Module):
#     def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
#         super().__init__()

#         tokenizer_class = TRANSFORMERS_MAP[transformers][1]
#         model_class = TRANSFORMERS_MAP[transformers][0]
#         self.tokenizer = tokenizer_class.from_pretrained(pretrained)
#         self.model = model_class.from_pretrained(pretrained)
#         self.use_finetune = use_finetune
    
#     def get_tokenizer(self):
#         return self.tokenizer
    
#     # def from_text(self, text):
#     #     """
#     #     text: raw data
#     #     """
#     #     input_ids = self.get_id(text)
#     #     with torch.no_grad():
#     #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
#     #     return last_hidden_states.squeeze()
    
#     def forward(self, text):
#         """
#         text: (batch_size, 3, seq_len)
#         3: input_ids, input_mask, segment_ids
#         input_ids: input_ids,
#         input_mask: attention_mask,
#         segment_ids: token_type_ids
#         """
#         input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
#         if self.use_finetune:
#             last_hidden_states = self.model(input_ids=input_ids,
#                                             attention_mask=input_mask,
#                                             token_type_ids=segment_ids)[0]  # Models outputs are now tuples
#         else:
#             with torch.no_grad():
#                 last_hidden_states = self.model(input_ids=input_ids,
#                                                 attention_mask=input_mask,
#                                                 token_type_ids=segment_ids)[0]  # Models outputs are now tuples
#         return last_hidden_states



import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

# 支持 BERT 和 RoBERTa 两种 transformer
TRANSFORMERS_MAP = {
    'bert':    (BertModel,    BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}

class BertTextEncoder(nn.Module):
    def __init__(
        self,
        use_finetune: bool = False,
        transformers: str = 'bert',
        # 本地模型目录：请确认该目录下有 config.json / tokenizer.json(or vocab.txt) / 权重文件
        pretrained: str = '/home/guze/work/IMDer/pretrained/bert-base-uncased',
        cache_dir: str = './.hf_cache',
        local_files_only: bool = True
    ):
        super().__init__()

        model_class, tokenizer_class = TRANSFORMERS_MAP[transformers]

        # 只从本地目录加载，不联网
        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        self.model = model_class.from_pretrained(
            pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )

        self.use_finetune = use_finetune

    # 获取 tokenizer 方便外部做编码
    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        text : torch.Tensor
            shape = (batch, 3, seq_len)
            channel-0: input_ids
            channel-1: attention_mask
            channel-2: token_type_ids
        """
        input_ids      = text[:, 0, :].long()
        attention_mask = text[:, 1, :].long()
        token_type_ids = text[:, 2, :].long()

        if self.use_finetune:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

        # 返回 last_hidden_state
        return outputs.last_hidden_state
