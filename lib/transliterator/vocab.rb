
module Transliterator


  class Languages

    attr_accessor :SourceCHARS, :TargetCHARS, :SpecialSymbols

    def initialize(config)

        @SourceCHARS = config["transliteration"]["SOURCECHARS"]
        @TargetCHARS = config["transliteration"]["TARGETCHARS"]
        @SpecialSymbols = config["transliteration"]["SPECIALSYMBOLS"]

    end

    # 'a   a  a a'-> 'a a a a'
    def collapse_whitespace(txt)

      txt.gsub(/[[:space:]]+/, " ")

    end

    # basic normalisation & clean up
    def clean(txt)

      txt = collapse_whitespace(txt)
      # rm chars not within SourceCHARS
      txt.chars.select {|c| @SourceCHARS.include? c}.join()

    end

  end

end
