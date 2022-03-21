
module Transliterator

  module Farsi

    Farsi_CHARS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
    Special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    #def pad(arr, length, no_eos=False, no_sos=False):
    #return numpy.pad(arr, (0, length - len(arr)), constant_values="<pad>")


    #def prepare_line(seq_length, vocab, no_sos=False, no_eos=False):
    #return lambda line: list(map(
    #  lambda c: vocab[c],
    #  pad(list(line), seq_length, no_sos=no_sos, no_eos=no_eos)
    # ))

  end

  module Transliterate

    Transliterated_CHARS = "ACMSXZabcdefghijklmnopqrstuvwxyz"

    Special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

  end

end
