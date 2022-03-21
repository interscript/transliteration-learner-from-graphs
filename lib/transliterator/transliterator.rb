
require_relative "vocab"
require_relative "encoders"
require_relative "transformer"


module Transliterator

  class Transliterator

    #include Transformer

    def initialize(onnx_model_path, config)

      @onnx_model_path = onnx_model_path
      @config = config
      @transformers = get_transformers()
      @encoders = get_encoders()

    end


    def transliterate_file(path)

      texts = File.read(path).split("\n").map(&:strip)

      # process batches
      out_texts = []
      idx = 0
      while idx + @batch_size <= texts.length
        srcs = texts[idx...idx + @batch_size]
        out_texts += transliterate_text(srcs)
        idx += @batch_size
      end

      # process rest of data
      while idx < texts.length
        out_texts += [transliterate_text(texts[idx])]
        idx += 1
      end

      out_texts
    end


    def transliterate_text(txt)
      raise NotImplementedError
    end


    def transliterate_mocked_text(txt)

      # encode txt into array (int)
      src = @encoders.encode_src_txt(txt)

      dic_batch = mock_dic_batch()

      dic_out = run_transformer_batch(dic_batch)

      # decode int arrays back into strings
      decode_batch(dic_out)

    end


    def transliterate_batch(batch_data)
      src = @encoders.encode_src_batch(batch_data)
    end


    def mock_dic_batch()

      input_vocab_size = 72
      target_vocab_size = 66
      batch_size = 16
      source_length = 8 #, 16
      target_length = 8

      src = Torch.randint(
        0, input_vocab_size, [source_length, batch_size])
      tgt = Torch.randint(
        0, target_vocab_size, [target_length, batch_size])
      tgt_mask = Torch.randint(
        0, 2, [target_length, target_length]).bool()
      src_key_padding_mask = Torch.randint(
          0, 2, [batch_size, source_length]).bool()
      tgt_key_padding_mask = Torch.randint(
          0, 2, [batch_size, target_length]).bool()
      memory_key_padding_mask = Torch.randint(
          0, 2, [batch_size, source_length]).bool()

      {"src" => src,
       "tgt" => tgt,
       "tgt_mask" => tgt_mask,
       "src_key_padding_mask" => src_key_padding_mask,
       "tgt_key_padding_mask" => tgt_key_padding_mask,
       "memory_key_padding_mask" => memory_key_padding_mask}

    end

    def run_transformer_batch(dic_batch)
      @transformers.onnx_session.predict(dic_batch)
    end

    def decode_batch(dic_out)

      out = dic_out["output"]

      # loop batch
      batch_size = 16
      max_len = 8

      vv = (0..batch_size-1).each.map do |j|
        (0..max_len-1).each.map do |i|
          id = out[i][j].map {|x| x.nan? ? -10 : x}.each_with_index.max[1]
          @encoders.target_id_to_s[id]
        end
      end

      # return transliterated strings
      vv.map do |v|
        v.join("")
      end

    end


    def get_transformers()
      Transformer.new(@onnx_model_path, @config)
    end


    def get_encoders()
      FarsiEncoder.new(@config)
    end

  end

end
