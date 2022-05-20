
require_relative "vocab"


module Transliterator

  class Encoder

    def initialize(config)

      @max_len = config["transliteration"]["max_str_len"]

      vocab_path = config["transliteration"]["VOCAB_TRAFO"]
      @vocab_transform = YAML.load(File.read(vocab_path))

      counter_path = config["transliteration"]["VOCAB_COUNTER"]
      @vocab_counter = YAML.load(File.read(counter_path))

      src_lang = config["transliteration"]["SRC_LAN"]
      tgt_lang = config["transliteration"]["TGT_LAN"]

      # maps id <-> wrd
      @src_s_to_id = @vocab_transform[src_lang].map.with_index { |s, i| [s, i] }.to_h
      @src_id_to_s = @vocab_transform[src_lang].map.with_index { |s, i| [i, s] }.to_h
      @tgt_s_to_id = @vocab_transform[tgt_lang].map.with_index { |s, i| [s, i] }.to_h
      @tgt_id_to_s = @vocab_transform[tgt_lang].map.with_index { |s, i| [i, s] }.to_h

      # map wrd -> cnt (aligned with above idces)
      @src_vocab_counter = @src_s_to_id.keys.map { |k| [k, @vocab_counter[k]] }.to_h

    end


    def tokenizer(txt)

      # clean and collapse whitespaces
      txt = txt.gsub(/[[:space:]]+/, " ").strip
      txt.split()

    end


    def encode_src(txt)

      l_txt = tokenizer(txt)
      src = l_txt.map { |wrd| [@src_s_to_id[wrd]] }
      src = []
      l_txt.map { |wrd|
         id = @src_s_to_id.fetch(wrd, false)
         # if string found, return it
         if id
           src += [[id]]
        # if not found, decompose string
         else
           src += encoder_unrecognised_wrd(wrd)
         end
      }
      # encoding beginning and ending
      [[2]] + src + [[3]]

    end


    def decode_tgt(tgt)

      txt = ""
      (1..tgt.length-2).map {|i| txt += @tgt_id_to_s[tgt[i]] + " "}
      txt.strip

    end


    def encoder_unrecognised_wrd(wrd)

      n_char = wrd.length
      score = 0

      # wrd represented by two strings
      model = [@src_s_to_id.fetch(wrd, 0), @src_s_to_id.fetch(wrd, 0)]

      (1..n_char).map { |i|
        # substrings
        w_1, w_2 = wrd[..i-1], wrd[i..]

        # substrings indices
        i_1, i_2 = @src_s_to_id.fetch(w_1, false),
                            @src_s_to_id.fetch(w_2, false)

        if i_1 && i_2 # if both found, then postprocess
          # score
          s = w_1.length * w_2.length
          # update wrd model if score larger
          if score < s
            score = s
            model = [[i_1], [i_2]]
          end
        end
      }

      # return chosen model with empty space
      if model != [0, 0]
        model
      else
        []
      end

    end


  end

end
