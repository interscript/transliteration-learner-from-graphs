
require_relative "vocab"

require "torch"

module Transliterator

  class FarsiEncoder

        attr_accessor :target_id_to_s
        # :normalized_table, :dagesh_table
        # include CharacterTable
        include Farsi
        include Transliterate

        def initialize(config)

          @max_len = config[:max_len]

          @input_s_to_id =
            (Special_symbols+Farsi_CHARS.chars).map.with_index { |s, i| [s, i] }.to_h
          @input_id_to_s =
            (Special_symbols+Farsi_CHARS.chars).map.with_index { |s, i| [i, s] }.to_h

          @target_s_to_id =
            (Special_symbols+Transliterated_CHARS.chars).map.with_index { |s, i| [s, i] }.to_h
          @target_id_to_s =
            (Special_symbols+Transliterated_CHARS.chars).map.with_index { |s, i| [i, s] }.to_h

        end

        def encode_src_txt(txt)

          src = txt.chars.map {|c| @input_s_to_id[c]}
          src = src + (1..@max_len-src.length).map  {|c| @input_s_to_id["<unk>"]}
          src = Torch.tensor(src)
          src
        end

        def encode_src_batch(batch_data)
          batch_data.map {|txt| encode_src_txt(txt)}
        end


        def transliterate_file(path)
          texts = File.read(path).split("\n").map(&:strip)

          # process batches
          out_texts = []
          idx = 0
          while idx + @batch_size <= texts.length
            originals = texts[idx...idx + @batch_size]

            out_texts += diacritize_text(originals)

            idx += @batch_size
          end

          # process rest of data
          while idx < texts.length
            out_texts += [diacritize_text(texts[idx])]
            idx += 1
          end

          out_texts
        end

  end

  class PositionalEncoding < Torch::NN::Module
    # PositionalEncoding module injects some information about the relative or
    # absolute position of the tokens in the sequence.
    # The positional encodings have the same dimension as the embeddings so
    # that the two can be summed. Here, we use sine and cosine functions of
    # different frequencies.
    def initialize(d_model, dropout: 0.1, max_len: 100) #5000)
      super()
      @dropout = Torch::NN::Dropout.new(p: dropout)

      pe = Torch.zeros(max_len, d_model)
      position = Torch.arange(0, max_len, dtype: :float).unsqueeze(1)
      div_term = Torch.exp(Torch.arange(0, d_model, 2).float() * (-Math.log(10000.0) / d_model))
      sin = Torch.sin(position * div_term).t
      cos = Torch.cos(position * div_term).t
      pe.t!
      pe.each.with_index do |row, i|
        pe[i] = sin[i / 2] if i % 2 == 0
        pe[i] = cos[(i-1)/2] if i % 2 != 0
      end
      pe.t!
      pe = pe.unsqueeze(0).transpose(0, 1)
      register_buffer('pe', pe)
    end

    def forward(x)
      x = x + pe.narrow(0, 0, x.size(0))
      return x
    end

  end

end
