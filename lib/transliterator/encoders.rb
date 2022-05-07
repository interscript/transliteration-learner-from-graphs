
require_relative "vocab"


module Transliterator

  class Encoder

    def initialize(config)

      @max_len = config["transliteration"]["max_str_len"]

      vocab_path = config["transliteration"]["VOCAB_TRAFO"]
      @vocab_transform = YAML.load(File.read(vocab_path))

      src_lang = config["transliteration"]["SRC_LAN"]
      tgt_lang = config["transliteration"]["TGT_LAN"]

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
      [[2]] + src + [[3]]

    end


    def decode_tgt(tgt)

      txt = ""
      (1..tgt.length-2).map {|i| txt += @tgt_id_to_s[tgt[i]] + " "}
      txt.strip

    end

  end

end
