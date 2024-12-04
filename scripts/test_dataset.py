
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.datasets import text_completion_dataset

def test_completion1():

   m_tokenizer = llama3_tokenizer(
       path="/workspace/.cache/tune/Meta-Llama-3.2-1B/original/tokenizer.model",
       max_seq_len=2048
   )

   ds = text_completion_dataset(
       tokenizer=m_tokenizer,
       source="bigcode/the-stack-smol",
       column="content",
    #    data_dir="data/python",
       split="train",
   )
   import pdb; pdb.set_trace()
   tokenized_dict = ds[0]
   print(m_tokenizer.decode(tokenized_dict["tokens"]))
   # After we were clear of the river Oceanus, and had got out into the open sea,\
   # we went on till we reached the Aeaean island where there is dawn and sunrise \
   # as in other places. We then drew our ship on to the sands and got out of her on \
   # to the shore, where we went to sleep and waited till day should break.
   print(tokenized_dict["labels"])
   # [128000, 6153, 584, 1051, 2867, 315, 279, 15140, 22302, 355, 11, 323, 1047, \
   # 2751, 704, 1139, 279, 1825, 9581, 11, 584, 4024, 389, 12222, 584, 8813, 279, \
   # 362, 12791, 5420, 13218, 1405, 1070, 374, 39493, 323, 64919, 439, 304, 1023, \
   # 7634, 13, 1226, 1243, 24465, 1057, 8448, 389, 311, 279, 70163, 323, 2751, 704, \
   # 315, 1077, 389, 311, 279, 31284, 11, 1405, 584, 4024, 311, 6212, 323, 30315, \
   # 12222, 1938, 1288, 1464, 13, 128001]

if __name__ == "__main__":
    test_completion1()