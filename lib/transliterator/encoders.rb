

module Transliterator

  class Encoder

    def initialize(config)

      @block_size = config["nnets"]["block_size"]
      chars_file_path = config["file_paths"]["chars_file_path"]
      @chars = File.read(chars_file_path).chars
      # maps id <-> wrd
      @map_c_to_id = @chars.map.with_index { |s, i| [s, i] }.to_h
      @map_id_to_c = @chars.map.with_index { |s, i| [i, s] }.to_h

    end


    def tokenizer(txt)

      txt.chars

    end


    def encode_src(txt)

      l_chars = tokenizer(txt)
      src = l_chars.map { |c| [@map_c_to_id[c]] }
      src = []
      l_chars.map { |c|
         id = @map_c_to_id.fetch(c, false)
         # if string found, return it
         if id
           src += [id]
         end
      }
      # encoding beginning and ending
      [src.fill(0, src.length, @block_size - src.length)]

    end


    def decode_tgt(tgt)

      tgt = tgt[0].map {|slice| slice.each_with_index.max[1]}
      txt = ""
      (1..tgt.length-2).map {|i| txt += @map_id_to_c[tgt[i]]}
      txt.strip

    end

  end

end
