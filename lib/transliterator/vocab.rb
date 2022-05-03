
module Transliterator


  class Languages

    attr_accessor :SourceCHARS, :TargetCHARS, :SpecialSymbols

    def initialize(config)

        @SourceCHARS = config["SourceCHARS"]
        @TargetCHARS = config["TargetCHARS"]
        @SpecialSymbols = config["SpecialSymbols"]

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
