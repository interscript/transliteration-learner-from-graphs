
require_relative "vocab"

require "torch"

module Transliterator

  class FarsiEncoder

    #include Farsi
    #include Transliterate

    def initialize(config)

      @max_len = config[:max_len]
      @vocab_transform = YAML.load(File.read(config[:vocab_transform]))

      src_lang = "farsi"
      tgt_lang = "transliterated"

      @src_s_to_id = @vocab_transform[src_lang].map.with_index { |s, i| [s, i] }.to_h
      @src_id_to_s = @vocab_transform[src_lang].map.with_index { |s, i| [i, s] }.to_h

      @tgt_s_to_id = @vocab_transform[tgt_lang].map.with_index { |s, i| [s, i] }.to_h
      @tgt_id_to_s = @vocab_transform[tgt_lang].map.with_index { |s, i| [i, s] }.to_h

    end

    def tokenizer(txt)

      # clean and collapse whitespaces
      txt = txt.gsub(/[[:space:]]+/, " ").strip
      txt.split()

    end

    def encode_src(txt)

      l_txt = tokenizer(txt)
      src = l_txt.map {|i| [@src_s_to_id[i]]}

      # encoding beginning and ending
      Torch.tensor([[2]] + src + [[3]])

    end

    def decode_tgt(tgt)

      txt = ""
      (1..tgt.length-2).map {|i| txt += @tgt_id_to_s[tgt[i]] + " "}
      txt.strip

    end

  end

end
