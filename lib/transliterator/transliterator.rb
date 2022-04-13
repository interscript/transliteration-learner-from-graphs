
require_relative "vocab"
require_relative "encoders"
require_relative "transformer"


module Transliterator

  class Transliterator


    def initialize(config)

      @config = config

      @transformers = get_transformers()
      @encoders = get_encoders()

    end


    def transliterate_file(path)

      texts = File.read(path).split("\n").map(&:strip)
      texts.map {|t| transliterate_text(t)}

    end


    def transliterate_text(txt)

      # transliteration steps with error handling
      #begin

        src = @encoders.encode_src(txt)
        tgt = @transformers.greedy_decode(src)
        str = @encoders.decode_tgt(tgt)
        str

      #rescue

      #  p("error processing string: " + txt)
      #  txt

      #end

    end


    def get_transformers()

      Transformer.new(@config)

    end


    def get_encoders()

      FarsiEncoder.new(@config)

    end

  end

end
